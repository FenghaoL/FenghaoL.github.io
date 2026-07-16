---
layout: about
title: about
permalink: /
subtitle: Robotics graduate student at the University of Michigan.

profile:
  align: right
  image: AlbertLiu.jpg
  image_circular: false
  more_info: >
    <p>Ann Arbor, MI</p>
    <p><a href="https://umich.edu/">University of Michigan</a></p>

selected_papers: false
social: false

announcements:
  enabled: false
  scrollable: true
  limit: 5

latest_posts:
  enabled: false
  scrollable: true
  limit: 3
---

<style>
  .home-projects {
    clear: both;
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 1.25rem;
    margin: 1.25rem 0 0;
  }

  .home-project-card {
    overflow: hidden;
    border: 1px solid var(--global-divider-color);
    border-radius: 8px;
    background: var(--global-bg-color);
  }

  .home-project-card video {
    display: block;
    width: 100%;
    aspect-ratio: 16 / 10;
    object-fit: cover;
    background: #111;
  }

  .home-project-card-body {
    padding: 1rem;
  }

  .home-project-card h2 {
    margin: 0 0 0.35rem;
    font-size: 1.15rem;
    line-height: 1.25;
  }

  .home-project-card h2 a {
    color: inherit;
  }

  .home-project-card p {
    margin: 0;
    color: var(--global-text-color-light);
    font-size: 0.95rem;
    line-height: 1.45;
  }

  .home-section-title {
    clear: both;
    margin-top: 2rem;
  }
</style>

I completed my undergraduate studies at [Southeast University](https://www.seu.edu.cn/english/) and am now a robotics graduate student at the [University of Michigan](https://umich.edu/).

I work on robot learning systems that connect large vision-language-action models with real robot data, post-training, and deployment. My recent work centers on SO101 manipulation: collecting teleoperated and autonomous rollouts, adapting VLA action spaces, and improving real-world policy behavior through SFT and offline preference optimization.

I am especially interested in practical embodied AI: data pipelines, action representations, policy evaluation, and the engineering details that make robot demos reproducible rather than lucky. I also have research experience in neural rendering and 3D reconstruction from sparse sonar data.

A concise summary of my experience is available in my [CV](/assets/pdf/FenghaoLiu_CV.pdf).

<h2 class="home-section-title">Selected Projects</h2>

<div class="home-projects">
  <div class="home-project-card">
    <a href="{{ '/projects/g05-so101-alignment/' | relative_url }}">
      <video muted playsinline preload="metadata" poster="{{ '/assets/img/projects/g05-cover.jpg' | relative_url }}">
        <source src="{{ '/assets/video/g0.5_sft_dpo/base_4epoch_sft/put-white-block-on-pink-bowl-19s.mp4' | relative_url }}" type="video/mp4">
      </video>
    </a>
    <div class="home-project-card-body">
      <h2><a href="{{ '/projects/g05-so101-alignment/' | relative_url }}">G0.5 Real-World SFT/RL Alignment on SO101</a></h2>
      <p>Real-world <a href="https://github.com/OpenGalaxea/GalaxeaVLA">G0.5</a> adaptation with 156 SFT demonstrations and window-DPO from success/failure rollouts.</p>
    </div>
  </div>

  <div class="home-project-card">
    <a href="{{ '/projects/pi05-libero-lora/' | relative_url }}">
      <video muted playsinline preload="metadata" poster="{{ '/assets/img/projects/pi05-cover.jpg' | relative_url }}">
        <source src="{{ '/assets/video/pi0.5_lora_full_compare/B_lora_hybrid_5999_libero_spatial_3trials_20260603_200135/rollout_pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_success.mp4' | relative_url }}" type="video/mp4">
      </video>
    </a>
    <div class="home-project-card-body">
      <h2><a href="{{ '/projects/pi05-libero-lora/' | relative_url }}">pi0.5 LoRA Ablations on LIBERO</a></h2>
      <p><a href="https://github.com/Physical-Intelligence/openpi">pi0.5</a> full fine-tuning and LoRA ablations across PaliGemma and action-expert modules on LIBERO-Spatial.</p>
    </div>
  </div>
</div>

<script>
  document.querySelectorAll(".home-project-card video").forEach((video) => {
    const card = video.closest(".home-project-card");
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
