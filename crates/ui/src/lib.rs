#![forbid(unsafe_code)]

//! `craftbio-ui` — Shared scientific UI layer for the CraftBio desktop suite, built directly on
//! upstream ArtCraft shared libraries (`craft-ui`, `craft-window`, `craft-dialogs`,
//! `craft-settings`, `craft-keymap`, `craft-i18n`).
//!
//! Provides:
//! - [`theme`]: unified dark/light scientific palettes built on [`craft_ui::theme::WidgetPalette`]
//!   with per-app brand accents and semantic status colors;
//! - [`prefs`]: persistent per-app UI preferences backed by [`craft_settings::SettingsStore`];
//! - [`window`]: consistent OS/Wayland/Hyprland/macOS/Windows frame policy via [`craft_window`];
//! - [`dialogs`]: non-blocking native file & folder pickers wired to [`egui::Context`] repaint;
//! - [`chrome`]: reusable scientific status bars, badges, cards and key-value grids for `egui 0.37`.

pub mod chrome;
pub mod dialogs;
pub mod prefs;
pub mod theme;
pub mod window;

pub use craft_dialogs;
pub use craft_i18n;
pub use craft_keymap;
pub use craft_settings;
pub use craft_ui;
pub use craft_window;
pub use egui;

pub use chrome::{badge, kv_grid, section_card, status_bar, theme_toggle_button};
pub use dialogs::{
    choose_folder, dialogs_for_egui, open_chemical_file, open_sequence_file, open_structure_file,
    save_file,
};
pub use prefs::{CommonUiPrefs, open_settings, titlebar_pref};
pub use theme::{AppAccent, CraftBioTheme, SemanticColors, ThemeMode};
pub use window::{WindowConfig, build_viewport};

#[cfg(test)]
mod tests {
    use super::*;
    use craft_dialogs::{Answer, Request};
    use craft_window::policy::{Backend, Desktop, DesktopKind, Frame, Os, TitleBarPref, decide};
    use std::path::PathBuf;
    use std::sync::Arc;

    #[test]
    fn theme_applies_dark_and_light_palettes_with_brand_accent() {
        let ctx = egui::Context::default();
        let dark = CraftBioTheme::new(ThemeMode::Dark, AppAccent::ADMET_EMERALD);
        dark.apply(&ctx);
        let dark_vis = ctx.global_style().visuals.clone();
        assert!(dark_vis.dark_mode);
        assert_eq!(
            dark_vis.selection.stroke.color,
            AppAccent::ADMET_EMERALD.color32()
        );

        let light = CraftBioTheme::new(ThemeMode::Light, AppAccent::BIOMOL_CYAN);
        light.apply(&ctx);
        let light_vis = ctx.global_style().visuals.clone();
        assert!(!light_vis.dark_mode);
        assert_eq!(
            light_vis.selection.stroke.color,
            AppAccent::BIOMOL_CYAN.color32()
        );
    }

    #[test]
    fn common_ui_prefs_defaults_and_serde_roundtrip() {
        let prefs = CommonUiPrefs::default();
        assert!(!prefs.dark_mode);
        assert!(!prefs.use_system_titlebar);
        assert_eq!(prefs.language, "es");
        assert!((prefs.font_scale - 1.0).abs() < f32::EPSILON);
        assert_eq!(
            titlebar_pref(prefs.use_system_titlebar),
            TitleBarPref::Integrated
        );
        assert_eq!(titlebar_pref(true), TitleBarPref::System);

        let json = serde_json::to_string(&prefs).unwrap_or_default();
        let parsed: CommonUiPrefs = serde_json::from_str(&json).unwrap_or_default();
        assert_eq!(parsed, prefs);
    }

    #[test]
    fn tiling_wayland_desktop_uses_native_frame_policy() {
        let tiling = Desktop {
            kind: DesktopKind::Tiling,
            ssd: Some(true),
            gtk_theme: None,
            icon_theme: None,
        };
        let decision = decide(
            Os::Linux,
            Backend::Wayland,
            &tiling,
            TitleBarPref::Integrated,
            None,
        );
        assert_eq!(decision.frame, Frame::Native);
    }

    #[test]
    fn non_blocking_dialogs_roundtrip_with_override() {
        let mut dialogs = craft_dialogs::Dialogs::<&'static str>::new(Arc::new(|| {}));
        dialogs.set_override(Some(Box::new(|req| match req {
            Request::File(_) => Answer::Paths(vec![PathBuf::from("molecules.smi")]),
            Request::Message(_) => Answer::Cancelled,
        })));
        dialogs.ask(open_chemical_file(None), "import");
        dialogs.show_pending::<wgpu_dummy::NoWindow>(None);
        let answers = dialogs.take_answers();
        assert_eq!(answers.len(), 1);
        assert_eq!(answers[0].0, "import");
        assert_eq!(
            answers[0].1,
            Answer::Paths(vec![PathBuf::from("molecules.smi")])
        );
    }

    mod wgpu_dummy {
        use raw_window_handle::{
            DisplayHandle, HandleError, HasDisplayHandle, HasWindowHandle, WindowHandle,
        };

        pub struct NoWindow;

        impl HasWindowHandle for NoWindow {
            fn window_handle(&self) -> Result<WindowHandle<'_>, HandleError> {
                Err(HandleError::Unavailable)
            }
        }

        impl HasDisplayHandle for NoWindow {
            fn display_handle(&self) -> Result<DisplayHandle<'_>, HandleError> {
                Err(HandleError::Unavailable)
            }
        }
    }
}
