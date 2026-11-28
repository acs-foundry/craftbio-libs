//! Unified window chrome & startup [`egui::ViewportBuilder`] setup via [`craft_window`].

use craft_window::Prefix;
use craft_window::policy::{Backend, Decision, TitleBarPref, apply, decide_for};

/// Window startup specification for a CraftBio desktop application.
#[derive(Debug, Clone, Copy)]
pub struct WindowConfig<'a> {
    pub title: &'a str,
    pub app_id: &'a str,
    pub env_prefix: &'static str,
    pub default_size: [f32; 2],
    pub min_size: [f32; 2],
    pub titlebar_pref: TitleBarPref,
}

impl<'a> WindowConfig<'a> {
    #[must_use]
    pub const fn new(title: &'a str, app_id: &'a str, env_prefix: &'static str) -> Self {
        Self {
            title,
            app_id,
            env_prefix,
            default_size: [1280.0, 820.0],
            min_size: [900.0, 600.0],
            titlebar_pref: TitleBarPref::Integrated,
        }
    }

    #[must_use]
    pub const fn with_size(mut self, default_size: [f32; 2], min_size: [f32; 2]) -> Self {
        self.default_size = default_size;
        self.min_size = min_size;
        self
    }

    #[must_use]
    pub const fn with_titlebar_pref(mut self, pref: TitleBarPref) -> Self {
        self.titlebar_pref = pref;
        self
    }
}

/// Probe the desktop session with [`craft_window::policy::decide_for`] and configure an
/// [`egui::ViewportBuilder`] consistently across Linux (Wayland/Hyprland/GNOME/KDE/X11), macOS
/// and Windows.
#[must_use]
pub fn build_viewport(cfg: &WindowConfig<'_>) -> (egui::ViewportBuilder, Decision) {
    let decision = decide_for(
        Prefix::new(cfg.env_prefix),
        Backend::from_env(),
        cfg.titlebar_pref,
    );
    let builder = egui::ViewportBuilder::default()
        .with_title(cfg.title)
        .with_app_id(cfg.app_id)
        .with_inner_size(cfg.default_size)
        .with_min_inner_size(cfg.min_size);
    let builder = apply(builder, &decision.frame);
    (builder, decision)
}
