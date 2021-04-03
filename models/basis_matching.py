import numpy as np
import torch
import torch.nn as nn
from backpack import backpack
from backpack.extensions import BatchGrad


def flatten_tensor(tensor_list):
    """Flatten and concat along non-batch dimensions."""
    return torch.cat(
        [t.flatten(start_dim=1) for t in tensor_list], dim=1
    )


@torch.jit.script
def orthogonalize(matrix):
    """Gram-Schmidt orthogonalization procedure."""
    n, m = matrix.shape
    for i in range(m):
        # Normalize the ith column.
        col = matrix[:, i: i + 1]
        col /= torch.sqrt(torch.sum(col ** 2))
        # Project it on the rest and remove it.
        if i + 1 < m:
            rest = matrix[:, i + 1:]
            rest -= torch.sum(col * rest, dim=0) * col


def clip_column(tsr, clip=1.0, inplace=True):
    if inplace:
        inplace_clipping(tsr, torch.tensor(clip).cuda())
    else:
        norms = torch.norm(tsr, dim=1)
        scale = torch.clamp(clip / norms, max=1.0)
        return tsr * scale.view(-1, 1)


@torch.jit.script
def inplace_clipping(matrix, clip):
    n, m = matrix.shape
    for i in range(n):
        # Normalize the i'th row
        col = matrix[i:i + 1, :]
        col_norm = torch.sqrt(torch.sum(col ** 2))
        if col_norm > clip:
            col /= (col_norm / clip)


def check_approx_error(L, target):
    """Compute the relative squared error."""
    encode = torch.matmul(target, L)  # n x k
    decode = torch.matmul(encode, L.t())
    error = torch.sum(torch.square(target - decode))
    target = torch.sum(torch.square(target))
    if target.item() == 0:
        return -1
    return error.item() / target.item()


def get_bases(pub_grad, num_bases, power_iter=1, logging=False):
    """QR algorithm for finding top-k eigenvalues.

    Returns:
        L: Tensor of selected basis of size (k, k).
        error_rate: Tensor of size (1,) for relative tolerance.
    """
    # The complexity (in non-parallelizable iteration count) is: power_iter * m
    num_p = pub_grad.shape[1]
    num_bases = min(num_bases, num_p)
    L = torch.normal(mean=0, std=1.0, size=(num_p, num_bases), device=pub_grad.device)
    for i in range(power_iter):
        R = torch.matmul(pub_grad, L)  # np, pk -> nk
        L = torch.matmul(pub_grad.t(), R)  # pn, nk -> pk
        orthogonalize(L)  # pk; orthonormalize the columns.
    error_rate = check_approx_error(L, pub_grad)
    return L, error_rate


class GEP(nn.Module):
    def __init__(self, num_bases, batch_size, clip0=1, clip1=1, power_iter=1):
        super(GEP, self).__init__()

        self.num_bases = num_bases  # The k for top-k subspace.
        self.clip0 = clip0
        self.clip1 = clip1
        self.power_iter = power_iter
        self.batch_size = batch_size
        self.approx_error = {}

    def get_approx_grad(self, embedding):
        bases_list, num_bases_list, num_param_list = (
            self.selected_bases_list, self.num_bases_list, self.num_param_list
        )
        grad_list = []
        offset = 0
        if len(embedding.shape) > 1:
            bs = embedding.shape[0]
        else:
            bs = 1
        embedding = embedding.view(bs, -1)

        for i, bases in enumerate(bases_list):
            num_bases = num_bases_list[i]
            grad = torch.matmul(
                embedding[:, offset:offset + num_bases].view(bs, -1), bases.T
            )
            if bs > 1:
                grad_list.append(grad.view(bs, -1))
            else:
                grad_list.append(grad.view(-1))
            offset += num_bases
        if bs > 1:
            return torch.cat(grad_list, dim=1)
        else:
            return torch.cat(grad_list)

    @torch.enable_grad()
    def get_anchor_gradients(self, net, loss_func):
        """Get the n x p matrix of gradients based on public data."""
        public_inputs, public_targets = self.public_inputs, self.public_targets
        outputs = net(public_inputs)
        loss = loss_func(outputs, public_targets)
        with backpack(BatchGrad()):
            loss.backward()
        cur_batch_grad_list = []
        for p in net.parameters():
            cur_batch_grad_list.append(p.grad_batch)
            del p.grad_batch
        return flatten_tensor(cur_batch_grad_list)  # n x p

    @torch.no_grad()
    def get_anchor_space(self, net, loss_func, logging=False):
        anchor_grads = self.get_anchor_gradients(net, loss_func)

        num_param_list = self.num_param_list

        selected_bases_list = []
        pub_errs = []

        # This is a parameter grouping heuristic detailed in Appendix B.
        # The motivation is to reduce the cost of power iteration. (mostly memory)
        sqrt_num_param_list = np.sqrt(np.array(num_param_list))
        num_bases_list = self.num_bases * (sqrt_num_param_list / np.sum(sqrt_num_param_list))
        num_bases_list = num_bases_list.astype(np.int)

        offset = 0
        for i, (num_param, num_bases) in enumerate(
            zip(num_param_list, num_bases_list)
        ):
            # Get the current group.
            pub_grad = anchor_grads[:, offset:offset + num_param]
            offset += num_param

            # Get the top eigen-space of this set of coordinates.
            selected_bases, pub_error = get_bases(
                pub_grad, num_bases, self.power_iter, logging
            )
            pub_errs.append(pub_error)
            selected_bases_list.append(selected_bases)

        self.selected_bases_list = selected_bases_list
        self.num_bases_list = num_bases_list
        self.approx_errors = pub_errs
        del anchor_grads

    @torch.no_grad()
    def forward(self, target_grad, logging=False):
        num_param_list = self.num_param_list
        embedding_list = []

        offset = 0
        if logging:
            print('group wise approx error')

        for i, num_param in enumerate(num_param_list):
            grad = target_grad[:, offset:offset + num_param]
            selected_bases = self.selected_bases_list[i]
            embedding = torch.matmul(grad, selected_bases)
            if logging:
                cur_approx = torch.matmul(torch.mean(embedding, dim=0).view(1, -1), selected_bases.T).view(-1)
                cur_target = torch.mean(grad, dim=0)
                cur_error = torch.sum(torch.square(cur_approx - cur_target)) / torch.sum(torch.square(cur_target))
                print('group %d, param: %d, num of bases: %d, group wise approx error: %.2f%%' % (
                    i, num_param, self.num_bases_list[i], 100 * cur_error.item()))
                if (i in self.approx_error):
                    self.approx_error[i].append(cur_error.item())
                else:
                    self.approx_error[i] = []
                    self.approx_error[i].append(cur_error.item())

            embedding_list.append(embedding)
            offset += num_param

        concatenated_embedding = torch.cat(embedding_list, dim=1)
        clipped_embedding = clip_column(concatenated_embedding, clip=self.clip0, inplace=False)
        if logging:
            norms = torch.norm(clipped_embedding, dim=1)
            print('average norm of clipped embedding: ', torch.mean(norms).item(), 'max norm: ',
                  torch.max(norms).item(), 'median norm: ', torch.median(norms).item())
        avg_clipped_embedding = torch.sum(clipped_embedding, dim=0) / self.batch_size

        no_reduction_approx = self.get_approx_grad(concatenated_embedding)
        residual_gradients = target_grad - no_reduction_approx
        clip_column(residual_gradients, clip=self.clip1)  # inplace clipping to save memory
        clipped_residual_gradients = residual_gradients
        if logging:
            norms = torch.norm(clipped_residual_gradients, dim=1)
            print('average norm of clipped residual gradients: ', torch.mean(norms).item(), 'max norm: ',
                  torch.max(norms).item(), 'median norm: ', torch.median(norms).item())

        avg_clipped_residual_gradients = torch.sum(clipped_residual_gradients, dim=0) / self.batch_size
        avg_target_grad = torch.sum(target_grad, dim=0) / self.batch_size
        return avg_clipped_embedding.view(-1), avg_clipped_residual_gradients.view(-1), avg_target_grad.view(-1)
