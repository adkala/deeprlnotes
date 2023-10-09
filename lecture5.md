$$
\def\at{\textbf{a}_t}
\def\st{\textbf{s}_t}
\def\ot{\textbf{o}_t}
\def\pt{\pi_\theta}
\def\M{\mathcal{M}}
\def\S{\mathcal{S}}
\def\T{\mathcal{T}}
\def\a{\textbf{a}}
\def\s{\textbf{s}}
\def\ptot{{\tau\sim p_\theta(\tau)}}
\def\jt{J(\theta)}
\def\pg{\nabla_\theta \jt}
\let\djt\pg
\def\prtt{p_\theta(\tau)}
\def\rt{r(\tau)}
\def\nt{\nabla_\theta}
$$

Addison Kalanther (addikala@berkeley.edu)

# Lecture 5: Policy Gradients

This lecture goes over the basics of policy gradients.

## Direct policy differentiation

The RL objective is taking the $\argmax$ of the expectation of the sum of rewards over the trajectory distribution of the policy. In policy differentiation, this is often estimated via sampling trajectories of a given policy and averaging their total rewards. This method of estimation is also known as _monte carlo_.

Defining $J(\theta) = E_\ptot[r(\tau)]$ as the RL objective, we can use the definition of expectation to evaluate this as $\jt = \int{p_\theta (\tau)r(\tau)d\tau}$ and take it's gradient to get the default policy gradient.

$$\djt = \int{\nabla_\theta p_\theta(\tau)r(\tau)d\tau} = \int{p_\theta(\tau)\nabla_\theta\log \prtt r(\tau)d\tau} = E_\ptot[\nabla\log\prtt \rt]$$

In the above derivation of the policy gradient, we use the identity $\prtt\nt \log \prtt = \prtt \frac{\nabla \prtt}{\prtt} = \nt\prtt$.

Using the above policy gradient, we can rewrite it using by expanding the $\log$ over the trajectory to get
$$\djt = E_\ptot[(\sum^T_{t=1}\nt\log\pt(\at \mid \st))(\sum^T_{t=1}r(\st, \at))] $$

The only difference between the two representations of the policy gradient $\pg$ is that $\nt\log\prtt$ and $\rt$ are expanded as sums over their individual states in a trajectory.

## Evaluating the policy gradient

We can find an approximation of the policy gradient using monte carlo approximation. Using the definition of the policy gradient $\pg$ found above, we can get its monte carlo approximation over $N$ samples as

$$\djt \approx \frac1{N}\sum^N_{i=1}(\sum^T_{t=1}\nt\log\pt(\at^i \mid \st^i))(\sum^T_{t=1}r(\st^i, \at^i))$$

This is the version of $\pg$ used in the REINFORCE algorithm, the first, most basic policy gradient.

### REINFORCE algorithm:

1. sample $\{\tau^i\}$ from $\pt(\at\mid\st)$ (run the policy)
2. $\djt \approx \sum_i (\sum_t \nt\log\pt(\at^i\mid\st^i))(\sum_t r(\st^i, \at^i))$
3. $\theta \leftarrow \theta + \alpha \pg$
4. repeat steps 1-3 until convergence

What the algorithm does is it first runs the policy to collect samples in step 1, calculates a monte carlo estimate of the policy gradient in step 2, and then uses that estimate to perform gradient ascent in step 3. It repeats this process until the policy converges.

## Understanding policy gradients

The policy gradient is essentially a weighted maximum likelihood using the total reward as weights. This means that the rate at which the probabilities of good actions goes up is faster than those of bad actions. This is obvious when comparing the equations of the two methods.

**Policy Gradient:** $$\djt \approx \frac1N \sum^N_{i=1}(\sum^T_{t=1}\nt\log\pt(\at^i\mid\st^i))(\sum^T_{t=1}r(\st^i, \at^i))$$

**Maximum Likelihood:** $$\nt J_{ML}(\theta) \approx \frac1N \sum^N_{i=1}(\sum^T_{t=1}\nt\log\pt(\at^i\mid\st^i))$$

## Example: gaussian policies

Assuming a gaussian policy,

$\pt(\at\mid\st) = \mathcal{N}(f_\text{nn}(\st); \Sigma)$,

we get $\log\pt(\at\mid\st) = -\frac12||f(\st) - \at||^2_\Sigma + \text{const}$

and $\nt\log\pt(\at\mid\st) = -\frac12\Sigma^{-1}(f(\st) - \at)\frac{df}{d\theta}$.

Using this, we can use REINFORCE to perform gradient ascent.

### What's going on here?

Optimizing the policy gradient just makes the good results more likely and the bad results less likely. This is ultimately a formalization of the notion of 'trial and error'. Try some actions and if it gives good results, make it more likely, if it does not, make it less likely than the better results.

#### Partial observability

Notice that all of this is done without assuming the markov property, this means that policy gradients can be used on partially observed processes as well.

### What is wrong with the policy gradient?

When using the raw total reward values in the policy gradient $\pg$, we can greatly push and pull the distributions if the values from the reward function have a large range of values and a high absolute value.

This means the variance of our policy could be very high. Even worse, if the good samples have a $\rt = 0$, this means that the distribution does not change at all for those samples.

## Reducing Variance

Here are a few methods we can use to reduce variance.

### Reward-to-go

We can reduce variance by finding a way to reduce the value of the 'weight' we are using. In default policy gradient, this is the reward. However, we can instead use a _reward-to-go_ rather than a total reward.

This is because of _causality_, the idea that a policy at time $t'$ cannot affect the reward at time $t$ when $t \lt t'$. We can use causality as the expected gradient of the reward $r(t)$ with respect to a policy at time t' is 0, since they would be uncorrelated. This leaves the policy gradient unbiased.

By taking advantage of causality, we ensure that better policies continue to be more probabilistic while reducing the magnitude of the total rewards for a given state, reducing overall variance of the policy.

### Baselines

We can further reduce the absolute value of reward by using a _baseline_ $b$. Using a baseline, our policy gradient $\pg$ becomes
$$\djt \approx \frac1N \sum^N_{i=1}\nt\log\prtt[\rt - b]$$

A convenient baseline is the average of total rewards over $N$ trajectories: $b = \frac1N \sum^N_{i=1}\rt$.

One question we should ask if we are allowed to do this. The condition we are allowed to do this in is if it leaves our policy gradient unbiased, we can show it does by showing that $E[\nt\log\prtt b] = 0$, since this would mean that our baselined policy is equal in expectation to the default policy by linearity of expectation.

We show that $E[\nt\log\prtt b] = 0$ by

$$E[\nt\log\prtt b] = \int{\prtt\nt\log\prtt b d\tau = \int{\nt\prtt b d\tau} = b\nt\int{\prtt d\tau} = b\nt1 = 0}$$

In the derivation, we used the convenient identity $\prtt\nt\log\prtt = \nt\prtt$

Average reward isn't the best baseline, to minimize variance, we can minimize via differentiation.

#### Optimal baseline

To minimize variance, we can take the derivative of the baselined variance with respect to b and set it to 0.

$$\text{Var} = E_\ptot[(\nt\log\prtt(\rt - b))^2] - E_\ptot[\nt\log\prtt(\rt - b)]^2$$

$$
\begin{aligned}
    \frac{d\text{Var}}{db} = \frac{d}{db}E[g(\tau)^2(\rt - b)^2] &= \frac{d}{db}(E[g(\tau)^2\rt^2] - 2E[g(\tau)^2\rt] + b^2E[g(\tau)^2])\\
    &= -2E[g(\tau)^2\rt] + 2bE[g(\tau)^2] = 0
\end{aligned}
$$

Solving the equation, we get $b = \frac{E[g(\tau)^2\rt]}{E[g(\tau)^2]}$. This can be interpreted as the expected reward weighted by gradient magnitudes.

## Off-policy policy gradients

Policy gradients according to the REINFORCE algorithm is on-policy. We have to resample after performing a gradient ascent step in order since the last samples don't come from the new policy's distribution, meaning the monte carlo expectation estimate of the new policy would not be correct unless we collect new samples.

This can be fixed using importance sampling.

### Importance sampling

#### Definition

Importance sampling allows us to calculate the expectation of one distribution using samples from the distribution of another.

Importance sampling allows us to rewrite the expectation of $f(x)$ over $p(x)$ as

$$E_{x\sim p(x)}[f(x)] = E_{x\sim q(x)}\left[\frac{p(x)}{q(x)}f(x)\right]$$

whose derivation follows simply from the definition of expectation.

#### Objective with importance sampling

The RL objective with importance sampling becomes

$$J(\theta) = E_{\tau\sim\bar{p}(\tau)}\left[\frac{\prtt}{\bar{p}(\tau)}\rt\right]$$

We can write the fractional term as

$$
\begin{aligned}
    \frac{\prtt}{\bar{p}(\tau)} &= \frac{p(\bold{s}_1)\prod_{t=1}^T\pt(\at\mid\st)p(\bold{s}_{t+1}\mid\st, \at)}{p(\bold{s}_1)\prod_{t=1}^T\bar{\pi}(\at\mid\st)p(\bold{s}_{t+1}\mid\st, \at)}\\
    &= \frac{\prod^T_{t=1}\pt(\at\mid\st)}{\prod^T_{t=1}\bar{\pi}(\at\mid\st)}
\end{aligned}
$$

#### Deriving the policy gradient with importance sampling

Using the importance sampled object, we get the policy gradient to be

$$\nabla_{\theta'}J(\theta') = E_\ptot\left[\frac{\nabla_{\theta'}p_{\theta'}(\tau)}{\prtt}\rt\right] = E_\ptot\left[\frac{p_{\theta'}(\tau)}{\prtt}\nabla_{\theta'}\log p_{\theta'}(\tau)\rt\right]$$

Estimated locally, at $\theta = \theta'$, we can see that it is equal to the policy gradient.

### The off-policy policy gradient

Expanding policy gradient with importance sampling when $\theta \neq \theta'$, we get

$$\nabla_{\theta'}J(\theta') = E_\ptot\left[\frac{p_{\theta'}(\tau)}{\prtt}\nabla_{\theta'}\log\pi_{\theta'}r(\tau)\right]$$

Substituting the per-state expansions for all three terms,

$$\nabla_{\theta'}J(\theta') = E_\ptot\left[\left(\prod^T_{t=1}\frac{\pi_{\theta'}(\at\mid\st)}{\pt(\at\mid\st)}\right) \left(\sum^T_{t=1}\nabla_{\theta'}\log\pi_{\theta'}(\at\mid\st)\right) \left(\sum^T_{t=1}r(\st, \at) \right) \right]$$

Accounting for causality (reward-to-go),

$$
\nabla_{\theta'}J(\theta') = E_\ptot\left[
    \sum^T_{t=1}\nabla_{\theta'}\log\pi_{\theta'}(\at\mid\st)
    \underline{\left(\prod^t_{t'=1}\frac{\pi_{\theta'}(\bold{a}_{t'}\mid\bold{s}_{t'})}{\pt(\bold{a}_{t'}\mid\bold{s}_{t'})}\right)}
    \left(\sum^T_{t'=t}r(\bold{s}_{t'}, \bold{a}_{t'})
        \cancel{\left(\prod^{t'}_{t''=t}\frac{\pi_{\theta'}(\bold{a}_{t''}\mid\bold{s}_{t''})}{\pt(\bold{a}_{t''}\mid\bold{s}_{t''})}\right)}
    \right)\right
]
$$

The underlined term takes into account the probabilities up until term $t$, this enforces causality as the probabilities of future actions shouldn't affect the current weight.

The crossed-out term is the probability that we get that specific total reward from the current state. However, this is numerically unstable so it is canceled out (prone to growing exponentially large). While this means that the equation with the canceled term is no longer the gradient, it can still be used to get an algorithm that converges to the optimal policy.

#### Aside: policy iteration

The equation with the crossed-out term is equivalent to policy iteration. This is because we take the probability that it gets to that state given its trajectory multiplied by the reward-to-go by that state.

If you are familiar with policy iteration, you will see that the sum over all trajectories of this value for a given state gives us the value function $V(\st)$ for that state. We use that value function to drive the rate at which the gradient moves, rather than just the reward of that specific rollout. The use of an implicit value function equates this equation to policy iteration.

#### A first-order approximation for IS

Our IS policy gradient is now

$$
\nabla_{\theta'}J(\theta') = E_\ptot\left[
    \sum^T_{t=1}\nabla_{\theta'}\log\pi_{\theta'}(\at\mid\st)
    \left(\prod^t_{t'=1}\frac{\pi_{\theta'}(\bold{a}_{t'}\mid\bold{s}_{t'})}{\pt(\bold{a}_{t'}\mid\bold{s}_{t'})}\right)
    \left(\sum^T_{t'=t}r(\bold{s}_{t'}, \bold{a}_{t'})\right)\right
]
$$

The middle term is exponential in T, we can approximate the gradient using it's first order approximation. We approximate the middle term using the probability of getting to state $\st$ and taking $\at$.

$$
\begin{aligned}
    \nabla_{\theta'}J(\theta') &\approx \frac1N\sum^N_{i=1}\sum^T_{t=1}\frac{\pi_{\theta'}(\st^i, \at^i)}{\pt(\st^i, \at^i)}\nabla_{\theta'}\log\pi_{\theta'}(\at^i\mid\st^i)\hat{Q}_{t}^i \\
    &= \frac1N\sum^N_{i=1}\sum^T_{t=1}\cancel{\frac{\pi_{\theta'}(\st^i)}{\pt(\st^i)}}\frac{\pi_{\theta'}(\at^i\mid\st^i)}{\pt(\at^i\mid\st^i)}\nabla_{\theta'}\log\pi_{\theta'}(\at^i\mid\st^i)\hat{Q}^i_t
\end{aligned}
$$

We can cross out that term because the policy update is not dependent on the probability of reaching that state and is rather dependent on the probability of taking a certain action given that state (since the policy output an action given a state, not the state itself). This will become more clear in the policy gradient lecture.

## Implementing policy gradients

We can implement policy gradients using automatic differentiation. Let's take a look at the policy gradient function.

$$\djt \approx \frac1N\sum^N_{i=1}\sum^T_{t=1}\nt\log\pt(\at^i\mid\st^i)\hat{Q}_t^i$$

The first term looks like the maximum likelihood gradient ($\nt J_{\text{ML}}(\theta) \approx \frac1N\sum^N_{i=1}\sum^T_{t=1}\nt\log\pt(\at^i\mid\st^i)$). The maximum likelihood is $J_{\text{ML}}(\theta) \approx \frac1N\sum^N_{i=1}\sum^T_{t=1}\log\pt(\at^i\mid\st^i)$.

Maximum likelihood is a loss function commonly computed in autodif packages. We can calculate the per timestep maximum likelihood loss and multiply it by the individual state-action values. Then, we can sum the weighted timesteps and divide it by the number of trajectories to recover the policy gradient.

### Policy gradient in practice

Remember policy gradient has a high variance, as this isn't the same as supervised learning. Gradients will be _very_ noisy.

Consider using much larger batches to reduce variance.

Tweaking learning rates can be very hard. Adaptive step size rules like ADAM can be ok, but policy gradient-specific learning rate adjustments will be discussed later.

## Advanced policy gradients

An issue with
