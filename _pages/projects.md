---
layout: page
title: projects
permalink: /projects/
description: Selected robot learning work.
nav: true
nav_order: 2
---

<style>
  .project-showcase {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 1.25rem;
    margin-top: 1.5rem;
  }

  .project-card {
    overflow: hidden;
    border: 1px solid var(--global-divider-color);
    border-radius: 8px;
    background: var(--global-bg-color);
  }

  .project-card video {
    display: block;
    width: 100%;
    aspect-ratio: 16 / 10;
    object-fit: cover;
    background: #111;
  }

  .project-card-body {
    padding: 1rem;
  }

  .project-card h2 {
    margin: 0 0 0.35rem;
    font-size: 1.15rem;
    line-height: 1.25;
  }

  .project-card p {
    margin: 0;
    color: var(--global-text-color-light);
    font-size: 0.95rem;
    line-height: 1.45;
  }
</style>

<div class="project-showcase">
  <a class="project-card" href="{{ '/projects/g05-so101-alignment/' | relative_url }}">
    <video muted playsinline preload="metadata" poster="{{ '/assets/img/projects/g05-cover.jpg' | relative_url }}">
      <source src="{{ '/assets/video/g0.5_sft_dpo/base_4epoch_sft/put-white-block-on-pink-bowl-19s.mp4' | relative_url }}" type="video/mp4">
    </video>
    <div class="project-card-body">
      <h2>G0.5 Real-World SFT/RL Alignment on SO101</h2>
      <p>4-epoch SFT on 156 demonstrations, followed by 1-epoch window-DPO from success/failure rollouts.</p>
    </div>
  </a>

  <a class="project-card" href="{{ '/projects/pi05-libero-lora/' | relative_url }}">
    <video muted playsinline preload="metadata" poster="{{ '/assets/img/projects/pi05-cover.jpg' | relative_url }}">
      <source src="{{ '/assets/video/pi0.5_lora_full_compare/B_lora_hybrid_5999_libero_spatial_3trials_20260603_200135/rollout_pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_success.mp4' | relative_url }}" type="video/mp4">
    </video>
    <div class="project-card-body">
      <h2>pi0.5 LoRA Ablations on LIBERO</h2>
      <p>Full fine-tuning and LoRA ablations across PaliGemma and action-expert modules on LIBERO-Spatial.</p>
    </div>
  </a>
</div>

<script>
  document.querySelectorAll(".project-card video").forEach((video) => {
    const card = video.closest(".project-card");
    const reset = () => {
      video.pause();
      video.currentTime = 0;
    };

    card.addEventListener("pointerenter", () => video.play());
    card.addEventListener("pointerleave", reset);
    card.addEventListener("focusin", () => video.play());
    card.addEventListener("focusout", reset);
  });
</script>
