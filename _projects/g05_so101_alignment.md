---
layout: page
title: G0.5 Real-World SFT/RL Alignment on SO101
description: SFT and offline preference optimization for a real SO101 manipulation policy.
img: assets/img/projects/g05-cover.jpg
importance: 1
category: work
permalink: /projects/g05-so101-alignment/
---

<style>
  .media-grid {
    display: grid;
    gap: 0.9rem;
    margin: 1.1rem 0 1.6rem;
  }

  .media-grid.two {
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  }

  .media-grid.three {
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  }

  .media-grid.five {
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  }

  .video-tile video,
  .wide-video {
    width: 100%;
    border-radius: 8px;
    background: #111;
  }

  .video-tile video {
    aspect-ratio: 4 / 3;
    object-fit: cover;
  }

  .wide-video {
    display: block;
  }

  .hero-video {
    width: min(70%, 760px);
    margin: 0 auto 1.3rem;
  }

  .compact-video {
    width: min(50%, 560px);
    margin: 1.1rem auto 0.35rem;
  }

  .video-caption {
    margin-top: 0.35rem;
    color: var(--global-text-color-light);
    font-size: 0.88rem;
    line-height: 1.35;
    text-align: center;
  }

  .dataset-caption {
    width: min(50%, 560px);
    margin: 0 auto 1.6rem;
  }

  .demo-label {
    margin: 1.2rem 0 0.35rem;
    color: var(--global-text-color-light);
    font-size: 0.95rem;
  }

  .stat-strip {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 0.75rem;
    margin: 1.2rem 0;
  }

  .stat-strip div {
    border: 1px solid var(--global-divider-color);
    border-radius: 8px;
    padding: 0.85rem;
  }

  .stat-strip strong {
    display: block;
    font-size: 1.35rem;
  }

  @media (max-width: 640px) {
    .hero-video,
    .compact-video,
    .dataset-caption {
      width: 100%;
    }
  }
</style>

<video class="wide-video hero-video" autoplay loop muted playsinline preload="metadata" poster="{{ '/assets/img/projects/g05-cover.jpg' | relative_url }}">
  <source src="{{ '/assets/video/g0.5_sft_dpo/base_4epoch_sft/put-white-block-on-pink-bowl-19s.mp4' | relative_url }}" type="video/mp4">
</video>

[G0.5](https://github.com/OpenGalaxea/GalaxeaVLA) is an autoregressive vision-language-action model: visual observations and language instructions are encoded into a token sequence, while robot actions are discretized through an ActionCodec-style action tokenizer and generated autoregressively. For SO101 deployment, I used the recent six-frame observation history as visual memory so the policy can condition on short-horizon motion context.

<div class="stat-strip">
  <div><strong>156</strong>SFT demonstration episodes</div>
  <div><strong>4</strong>training tasks</div>
  <div><strong>52</strong>usable preference rollouts</div>
  <div><strong>2,409</strong>window-DPO samples</div>
</div>

## Stage 1: SFT for Real-World Skills

I collected SO101 demonstrations for four tabletop manipulation instructions, covering pick-and-place behavior over the white block, paper roll, and pink bowl. After 4 epochs of SFT, the policy learned the training skills and also showed compositional generalization in follow-up evaluation: without demonstrations for "put white block on pink bowl," it could execute that new instruction accurately.

<video class="wide-video compact-video" autoplay loop muted playsinline preload="metadata">
  <source src="{{ '/assets/video/g0.5_sft_dpo/square_tasks_7x7.mp4' | relative_url }}" type="video/mp4">
</video>
<div class="video-caption dataset-caption">Collected SFT data: 156 SO101 demonstration episodes across four tabletop instructions.</div>

<p class="demo-label">Policy demos after the 4-epoch SFT checkpoint:</p>
<div class="media-grid five">
  <div class="video-tile">
    <video autoplay loop muted playsinline preload="metadata">
      <source src="{{ '/assets/video/g0.5_sft_dpo/base_4epoch_sft/pick_up_paper_roll.mp4' | relative_url }}" type="video/mp4">
    </video>
    <div class="video-caption">Pick up paper roll</div>
  </div>
  <div class="video-tile">
    <video autoplay loop muted playsinline preload="metadata">
      <source src="{{ '/assets/video/g0.5_sft_dpo/base_4epoch_sft/pick_up_white_block.mp4' | relative_url }}" type="video/mp4">
    </video>
    <div class="video-caption">Pick up white block</div>
  </div>
  <div class="video-tile">
    <video autoplay loop muted playsinline preload="metadata">
      <source src="{{ '/assets/video/g0.5_sft_dpo/base_4epoch_sft/put%20paper%20roll%20on%20pink%20bowl.mp4' | relative_url }}" type="video/mp4">
    </video>
    <div class="video-caption">Paper roll on pink bowl</div>
  </div>
  <div class="video-tile">
    <video autoplay loop muted playsinline preload="metadata">
      <source src="{{ '/assets/video/g0.5_sft_dpo/base_4epoch_sft/put%20white%20block%20on%20paper%20roll.mp4' | relative_url }}" type="video/mp4">
    </video>
    <div class="video-caption">White block on paper roll</div>
  </div>
  <div class="video-tile">
    <video autoplay loop muted playsinline preload="metadata">
      <source src="{{ '/assets/video/g0.5_sft_dpo/base_4epoch_sft/put%20white%20block%20on%20pink%20bowl.mp4' | relative_url }}" type="video/mp4">
    </video>
    <div class="video-caption">Unseen: white block on pink bowl</div>
  </div>
</div>

## Stage 2: Offline Preference Alignment

The 4-epoch SFT policy still failed under some edge initializations. I then collected autonomous successes, autonomous failures, and human-intervention rescue rollouts across eight initial-state buckets. Within each bucket, successful trajectories are treated as chosen samples and failed trajectories as rejected samples.

<video class="wide-video compact-video" autoplay loop muted playsinline preload="metadata">
  <source src="{{ '/assets/video/g0.5_sft_dpo/rl_bucket_7x7.mp4' | relative_url }}" type="video/mp4">
</video>
<div class="video-caption dataset-caption">Collected offline alignment data: autonomous successes, autonomous failures, and human-intervention rescue rollouts across eight initial-state buckets.</div>

For each same-instruction, same-bucket pair, I train on short trajectory windows rather than isolated frames. The objective compares the policy's action-token likelihood against the reference checkpoint:

$$
\begin{aligned}
L_{\mathrm{DPO}}
&=
-\log \sigma
\left(
\beta
\left[
(\log \pi_\theta(y^+) - \log \pi_{\mathrm{ref}}(y^+))
-
(\log \pi_\theta(y^-) - \log \pi_{\mathrm{ref}}(y^-))
\right]
\right).
\end{aligned}
$$

Here $y^+$ and $y^-$ are success/failure action-token windows. The final run uses 85 raw pairs, 78 trainable base pairs after indexing checks, window size 8, stride 4, $\beta=0.1$, and an additional chosen-trajectory CE term.

<div class="media-grid three">
  <div class="video-tile">
    <video autoplay loop muted playsinline preload="metadata">
      <source src="{{ '/assets/video/g0.5_sft_dpo/RL_base.mp4' | relative_url }}" type="video/mp4">
    </video>
    <div class="video-caption">4-epoch SFT base</div>
  </div>
  <div class="video-tile">
    <video autoplay loop muted playsinline preload="metadata">
      <source src="{{ '/assets/video/g0.5_sft_dpo/RL_sft.mp4' | relative_url }}" type="video/mp4">
    </video>
    <div class="video-caption">Success-only SFT +1 epoch</div>
  </div>
  <div class="video-tile">
    <video autoplay loop muted playsinline preload="metadata">
      <source src="{{ '/assets/video/g0.5_sft_dpo/RL_dpo.mp4' | relative_url }}" type="video/mp4">
    </video>
    <div class="video-caption">Window-DPO +1 epoch</div>
  </div>
</div>

The DPO-aligned policy succeeds on the difficult initialization where both the original SFT policy and the success-only SFT control remain unstable. In this setup, offline preference alignment gives faster convergence than simply adding one more epoch of positive demonstrations.
