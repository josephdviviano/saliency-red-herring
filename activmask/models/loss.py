import torch
import numpy as np
import torch.nn.functional as F


def compare_activations(act_a, act_b):
    """
    Calculates the mean l2 norm between the lists of activations
    act_a and act_b.
    """
    assert len(act_a) == len(act_b)
    dist = torch.nn.modules.distance.PairwiseDistance(p=2)
    all_dists = []

    # Gather all L2 distances between the activations
    for a, b in zip(act_a, act_b):
        all_dists.append(dist(a, b).view(-1))

    all_dists = torch.cat(all_dists)
    actdiff_loss = all_dists.sum() / len(all_dists)

    return(actdiff_loss)


def dim2margin(d, s=3.):
    """
    s is a hyperparameter which tunes how smooth the transition is from
    +1 to -1.
    """
    _base_var = 1.  # .25 is for 0-1 bernoulli, this is +/- bernoulli
    return(s*np.sqrt(_base_var / d))


def softsign(h, epsilon=1e-3):
    """
    A somewhat-smooth representation of a binarized vector, which allows
    for gradients to flow.
    """
    _mu = abs(h).mean()
    h_epsilon = np.float32(epsilon) * _mu
    h_epsilon = h_epsilon.detach()
    act = (h / (abs(h) + h_epsilon))

    return act


def get_bre_loss(model, epsilon=1e-3, binarizer="softsign", s=3.0):
    """
    Improving GAN training via binarized representation entropy regularization.
    Cao et al. ICLR 2018.
    """
    me_value, me_stats = me_term(
        model.all_activations, epsilon=epsilon, binarizer=binarizer)
    ac_value, ac_stats = ac_term(
        model.all_activations, epsilon=epsilon, binarizer=binarizer, s=s)
    bre_loss = me_value + ac_value

    return (bre_loss, me_value, ac_value, me_stats, ac_stats)


def me_term(hs, epsilon=1e-3, binarizer='softsign'):
    """
    Gets the marginal of the joint entropy of the binarized activations
    for each layer, averaging across layers.
    """
    h_me = []
    stats = []
    binarizer = globals()[binarizer]

    for h in hs:

        # Flatten each member of the minibatch.
        h = torch.flatten(h, 1)

        # Debug mean of absoloute activations.
        stats.append(abs(h).mean())
        act = binarizer(h, epsilon)

        # Mean of the i-th activations across the minibatch k, squared, then
        # mean of those i activations.
        h_me.append(torch.mean(act, 0).pow(2).mean())

    # Mean of the me term across all layers.
    hs_me = sum(h_me) /  np.float32(len(hs))
    stats = sum(stats) / np.float32(len(stats))

    return (hs_me, stats)


def ac_term(hs, epsilon=1e-3, s=3., binarizer='softsign'):
    """
    Get the mean correlation of the activation vectors.
    """
    assert s >= 1.

    C = 0.0
    stats_C = 0.0
    stats_abs_mean = []  # Keeps track of the mean(abs()) value.
    stats_sat90_ratio = []  # Keeps track of the % of values over abs(0.90).
    stats_sat99_ratio = []  # Keeps track of the % of values over abs(0.99).
    binarizer = globals()[binarizer]

    for h in hs:
        act = binarizer(h, epsilon)

        stats_abs_mean.append(torch.mean(abs(act)))
        stats_sat90_ratio.append(torch.mean((abs(act) > .90).double()))
        stats_sat99_ratio.append(torch.mean((abs(act) > .99).double()))

        act = torch.flatten(act, 1)

        # Dot product of activations with transpose.
        l_C = torch.mm(act, act.t()) / act.shape[1]
        stats_C += l_C

        # Store the max of (0, l_C) in C.
        C_tmp = abs(l_C) - dim2margin(act.shape[1], s)
        C_tmp[C_tmp < 0] = 0
        C += C_tmp

    # Calculate final mean correlation across all layers.
    C /= np.float32(len(hs))
    C -= torch.diag(torch.diag(C))
    C = torch.mean(C)

    # Calculate statistics for debugging and reporting.
    stats_abs_mean = sum(stats_abs_mean) / np.float32(len(hs))
    stats_sat90_ratio = sum(stats_sat90_ratio) / np.float32(len(hs))
    stats_sat99_ratio = sum(stats_sat99_ratio) / np.float32(len(hs))
    stats_C /= np.float32(len(hs))
    stats_Cmin = stats_C.min(1)[0]  # collect before we 0 the diagonal
    stats_C -= torch.diag(torch.diag(stats_C))
    stats_Cmean = stats_C.sum(0) / stats_C.shape[1]-1
    stats_Cmax = stats_C.max(0)

    stats = {"cmin": stats_Cmin,
             "cmean": stats_Cmean,
             "cmax": stats_Cmax,
             "abs_mean": stats_abs_mean,
             "sat90_ratio": stats_sat90_ratio,
             "sat99_ratio": stats_sat99_ratio}

    return (C, stats)


def get_grad_contrast(X, y_pred):
    """Gradmask: Simple Constrast Loss. d(y_0-y_1)/dx"""
    contrast = torch.abs(y_pred[:, 0] - y_pred[:, 1])
    # This is always a list of length 1, so we remove the element from the list.
    gradients = torch.autograd.grad(
        outputs=contrast, inputs=X, allow_unused=True, create_graph=True,
        grad_outputs=torch.ones_like(contrast))[0]

    return gradients


def get_grad_rrr(X, y_pred):
    """Right for the Right Reasons."""
    y_pred = torch.sum(torch.log(F.softmax(y_pred, dim=1)), 1)
    # This is always a list of length 1, so we remove the element from the list.
    gradients = torch.autograd.grad(
        outputs=y_pred, inputs=X, allow_unused=True, create_graph=True,
        grad_outputs=torch.ones_like(pos_class))[0]

    return gradients


def get_gradmask_loss(x, class_output, model, target, penalise_grad="contrast"):
    if penalise_grad == "contrast":
        # d(y_0-y_1)/dx
        input_grads = torch.autograd.grad(
            outputs = torch.abs(class_output[:, 0]-class_output[:, 1]).sum(),
            inputs=x, allow_unused=True, create_graph=True)[0]

    elif penalise_grad == "nonhealthy":
        # select the non healthy class d(y_1)/dx
        input_grads = torch.autograd.grad(
            outputs=torch.abs(class_output[:, 1]).sum(), inputs=x,
            allow_unused=True, create_graph=True)[0]

    elif penalise_grad == "diff_from_ref":
        # do the deep lift style ref update and diff-to-ref calculations here

        # update the reference stuff
        # 1) mask all_activations to get healthy only
        try:
            target = target.to(x.get_device())
        except:
            pass

        healthy_mask = target.float().reshape(-1, 1, 1, 1).clone()
        healthy_mask[target.float().reshape(-1, 1, 1, 1) == 0] = 1
        healthy_mask[target.float().reshape(-1, 1, 1, 1) != 0] = 0

        diff = torch.FloatTensor()

        try:
            diff = diff.to(x.get_device())
        except:
            pass

        # print("Activation lengths: ", len(model.all_activations))
        for i in range(len(model.all_activations)):
            a = model.all_activations[i]
            # print("Activation shape: ", a.shape)

            if len(a.shape) < 4:
                # activations from last layers (shape [batch, FC layer_size])
                new_mask = target.float().reshape(-1, 1)
                new_mask[target.float().reshape(-1, 1) == 0] = 1
                new_mask[target.float().reshape(-1, 1) != 0] = 0
                healthy_batch = new_mask * a
            else:
                healthy_batch = healthy_mask * a

            # 2) detach grads for the healthy samples
            healthy_batch = healthy_batch.detach()

            # 3) get batch-wise average of activations per layer
            batch_avg_healthy = torch.mean(healthy_batch, dim=0)

            # 4) update global reference average in model's deep_lift_ref attr
            if len(model.ref) < len(model.all_activations):
                # for the first iteration, just make the model.ref ==
                # batch_avg_healthy for that layer
                model.ref.append(batch_avg_healthy)
            else:
                # otherwise, a rolling average
                model.ref[i] = model.ref[i] * 0.8 + batch_avg_healthy

            # 5) TODO: somehow incorporate std to allowing regions of variance
            # in the healthy images use the reference layers to get the
            # diff-to-ref of each layer and output contribution scores
            # contribution scores should be the input_grads? Should be a single
            # matrix of values for how each input pixel contributes to the
            # output layer, no? Like all the layer-wise diff-from-ref get
            # condensed into one thing based on sum(contribution_scores of
            # (delta_x_i, delta_t)) = delta_t
            # 1) for each layer, t - t0, then mask the unhealthy ones
            # 2) flatten, 3) stack or join together somehow?, 4) L1 norm,
            # 5) input grads
            diff = torch.cat(
                (diff, torch.flatten(
                    (a - model.ref[i]) * target.float().reshape(-1, 1, 1, 1))
                )
            )

        input_grads = torch.autograd.grad(outputs=torch.abs(diff).sum(),
                                          inputs=x, allow_unused=True,
                                          create_graph=True)[0]

    elif penalise_grad == "masd_style":
        # In the style of the Model-Agnostic Saliency Detector paper:
        # https://arxiv.org/pdf/1807.07784.pdf
        print(class_output.shape, representation.shape)
        # outputs should now be: abs(diff(area_seg - area_saliency_map)) +
        # WIP
    else:
        raise Exception(
            "invalid penalise_grad: {contrast, nonhealthy, diff_from_ref}")

    return input_grads
