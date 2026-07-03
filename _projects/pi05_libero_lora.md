---
layout: page
title: pi0.5 Tuning Strategies on LIBERO
description: Full fine-tuning and LoRA ablations for pi0.5 on LIBERO-Spatial.
img: assets/img/projects/pi05-cover.jpg
importance: 2
category: work
permalink: /projects/pi05-libero-lora/
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

  .media-grid.five {
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  }

  .video-tile video,
  .wide-video,
  .project-img {
    width: 100%;
    border-radius: 8px;
    background: #111;
  }

  .video-tile video {
    aspect-ratio: 1 / 1;
    object-fit: cover;
  }

  .wide-video,
  .project-img {
    display: block;
  }

  .video-caption {
    margin-top: 0.35rem;
    color: var(--global-text-color-light);
    font-size: 0.88rem;
    line-height: 1.35;
    text-align: center;
  }

  .arch-diagram {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 0.65rem;
    margin: 1rem 0 1.4rem;
  }

  .arch-diagram div {
    border: 1px solid var(--global-divider-color);
    border-radius: 8px;
    padding: 0.85rem;
    text-align: center;
  }

  .arch-diagram strong {
    display: block;
    margin-bottom: 0.3rem;
  }

  .compact-table {
    width: 100%;
    margin: 1rem 0 1.4rem;
    font-size: 0.92rem;
  }

  .compact-table th,
  .compact-table td {
    border-bottom: 1px solid var(--global-divider-color);
    padding: 0.45rem 0.35rem;
    vertical-align: top;
  }
</style>

<video class="wide-video" autoplay loop muted playsinline preload="metadata" poster="{{ '/assets/img/projects/pi05-cover.jpg' | relative_url }}">
  <source src="{{ '/assets/video/pi0.5_lora_full_compare/B_lora_hybrid_5999_libero_spatial_3trials_20260603_200135/rollout_pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_success.mp4' | relative_url }}" type="video/mp4">
</video>

pi0.5 is a flow-matching VLA policy that generates continuous robot actions through an action expert conditioned on visual and language features. I reproduced the LIBERO-Spatial fine-tuning pipeline and compared which parts of the model should be adapted.

<div class="arch-diagram">
  <div><strong>Images</strong>SigLIP visual encoder</div>
  <div><strong>Language</strong>PaliGemma VLM backbone</div>
  <div><strong>Policy</strong>Action expert</div>
  <div><strong>Output</strong>Flow-matched action chunks</div>
</div>

## Ablation Setup

All variants were trained on 4 x A100 40GB, evaluated at checkpoint step 5999 with 3 trials per task, 10 tasks, and 30 total evaluation rollouts.

<table class="compact-table">
  <thead>
    <tr>
      <th>ID</th>
      <th>Configuration</th>
      <th>Trainable parameters</th>
      <th>Success</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>A</td><td>Full fine-tuning</td><td>~3.40B / 100%</td><td>29/30 = 96.7%</td></tr>
    <tr><td>B</td><td>Hybrid LoRA: PaliGemma LoRA + action expert LoRA + non-LLM full params</td><td>466.96M / 13.72%</td><td>25/30 = 83.3%</td></tr>
    <tr><td>C</td><td>PaliGemma LoRA + action expert LoRA</td><td>49.99M / 1.47%</td><td>14/30 = 46.7%</td></tr>
    <tr><td>D</td><td>Action expert LoRA only</td><td>22.12M / 0.65%</td><td>4/30 = 13.3%</td></tr>
    <tr><td>E</td><td>PaliGemma LoRA only</td><td>27.87M / 0.82%</td><td>13/30 = 43.3%</td></tr>
  </tbody>
</table>

<img class="project-img" src="{{ '/assets/img/projects/pi05-libero-ablation.png' | relative_url }}" alt="LIBERO-Spatial success rate ablation chart">

<div class="media-grid two">
  <img class="project-img" src="{{ '/assets/img/Wandb_pi0.5_lora.png' | relative_url }}" alt="WandB training loss curves for pi0.5 LoRA ablations">
  <div>
    <p>The training curves separate cleanly by adaptation capacity: full fine-tuning and hybrid LoRA converge fastest, while action-expert-only LoRA underfits the spatial instruction distribution.</p>
    <p>The strongest parameter-efficient setting is the hybrid LoRA preset, which preserves most of the full fine-tuning performance while training only 13.72% of the model.</p>
  </div>
</div>

## Same-Task Demo Comparison

The clips below use the same LIBERO instruction, "pick up the black bowl from table center and place it on the plate." A, B, C, and E use successful rollouts; D shows the available failure case.

<div class="media-grid five">
  <div class="video-tile">
    <video autoplay loop muted playsinline preload="metadata">
      <source src="{{ '/assets/video/pi0.5_lora_full_compare/A_full_finetune_5999_libero_spatial_3trials_20260603_200135/rollout_pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_success.mp4' | relative_url }}" type="video/mp4">
    </video>
    <div class="video-caption">A Full</div>
  </div>
  <div class="video-tile">
    <video autoplay loop muted playsinline preload="metadata">
      <source src="{{ '/assets/video/pi0.5_lora_full_compare/B_lora_hybrid_5999_libero_spatial_3trials_20260603_200135/rollout_pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_success.mp4' | relative_url }}" type="video/mp4">
    </video>
    <div class="video-caption">B Hybrid</div>
  </div>
  <div class="video-tile">
    <video autoplay loop muted playsinline preload="metadata">
      <source src="{{ '/assets/video/pi0.5_lora_full_compare/C_lora_pure_both_5999_libero_spatial_3trials_20260603_200135/rollout_pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_success.mp4' | relative_url }}" type="video/mp4">
    </video>
    <div class="video-caption">C PaliGemma + Action</div>
  </div>
  <div class="video-tile">
    <video autoplay loop muted playsinline preload="metadata">
      <source src="{{ '/assets/video/pi0.5_lora_full_compare/D_lora_pure_action_5999_libero_spatial_3trials_20260603_200135/rollout_pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_failure.mp4' | relative_url }}" type="video/mp4">
    </video>
    <div class="video-caption">D Action only</div>
  </div>
  <div class="video-tile">
    <video autoplay loop muted playsinline preload="metadata">
      <source src="{{ '/assets/video/pi0.5_lora_full_compare/E_lora_pure_pg_5999_libero_spatial_3trials_20260603_200135/rollout_pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_success.mp4' | relative_url }}" type="video/mp4">
    </video>
    <div class="video-caption">E PaliGemma only</div>
  </div>
</div>

The ranking is intuitive overall: adapting more of the model improves success. The surprising result is that PaliGemma-only LoRA outperforms action-expert-only LoRA. My interpretation is that LIBERO-Spatial requires adapting spatial-language grounding, which is partly represented in the SigLIP/PaliGemma visual-language stack. The action expert already starts from a policy prior over Cartesian end-effector actions, so tuning it alone provides limited benefit for this benchmark.
