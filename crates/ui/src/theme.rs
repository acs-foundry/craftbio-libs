//! Cohesive dark/light scientific themes built on [`craft_ui::theme::WidgetPalette`].

use craft_ui::theme::WidgetPalette;
use egui::{Color32, CornerRadius, Stroke, Visuals};
use serde::{Deserialize, Serialize};

/// Light or dark theme mode shared across CraftBio apps.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ThemeMode {
    #[default]
    Light,
    Dark,
}

impl ThemeMode {
    #[must_use]
    pub const fn from_dark(dark: bool) -> Self {
        if dark { Self::Dark } else { Self::Light }
    }

    #[must_use]
    pub const fn is_dark(self) -> bool {
        matches!(self, Self::Dark)
    }

    pub fn toggle(&mut self) {
        *self = match self {
            Self::Dark => Self::Light,
            Self::Light => Self::Dark,
        };
    }

    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Dark => "Dark",
            Self::Light => "Light",
        }
    }

    /// Label for a button that switches to the opposite mode.
    #[must_use]
    pub const fn switch_label(self) -> &'static str {
        match self {
            Self::Dark => "Light",
            Self::Light => "Dark",
        }
    }
}

/// Brand accent color for a CraftBio application.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AppAccent {
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

impl AppAccent {
    pub const ADMET_EMERALD: Self = Self::rgb(16, 185, 129);
    pub const BIOMOL_CYAN: Self = Self::rgb(14, 165, 233);
    pub const CHEM_AMBER: Self = Self::rgb(245, 158, 11);
    pub const DOCK_VIOLET: Self = Self::rgb(139, 92, 246);
    pub const PATH_CORAL: Self = Self::rgb(239, 68, 68);
    pub const SEQ_INDIGO: Self = Self::rgb(99, 102, 241);
    pub const GENE_TEAL: Self = Self::rgb(20, 184, 166);

    #[must_use]
    pub const fn rgb(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b }
    }

    #[must_use]
    pub const fn color32(self) -> Color32 {
        Color32::from_rgb(self.r, self.g, self.b)
    }

    #[must_use]
    pub fn soft_fill(self, dark: bool) -> Color32 {
        Color32::from_rgba_unmultiplied(self.r, self.g, self.b, if dark { 56 } else { 36 })
    }
}

/// Semantic scientific status colors calibrated for contrast on light and dark backgrounds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SemanticColors {
    pub ok: Color32,
    pub warn: Color32,
    pub danger: Color32,
    pub info: Color32,
    pub muted: Color32,
    pub card_stroke: Color32,
}

impl SemanticColors {
    #[must_use]
    pub const fn for_mode(mode: ThemeMode) -> Self {
        match mode {
            ThemeMode::Dark => Self {
                ok: Color32::from_rgb(52, 211, 153),
                warn: Color32::from_rgb(251, 191, 36),
                danger: Color32::from_rgb(248, 113, 113),
                info: Color32::from_rgb(56, 189, 248),
                muted: Color32::from_rgb(148, 163, 184),
                card_stroke: Color32::from_rgb(51, 65, 85),
            },
            ThemeMode::Light => Self {
                ok: Color32::from_rgb(5, 150, 105),
                warn: Color32::from_rgb(217, 119, 6),
                danger: Color32::from_rgb(220, 38, 38),
                info: Color32::from_rgb(2, 132, 199),
                muted: Color32::from_rgb(100, 116, 139),
                card_stroke: Color32::from_rgb(203, 213, 225),
            },
        }
    }
}

/// Complete CraftBio desktop theme specification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CraftBioTheme {
    pub mode: ThemeMode,
    pub accent: AppAccent,
    pub corner_radius: u8,
}

impl CraftBioTheme {
    #[must_use]
    pub const fn new(mode: ThemeMode, accent: AppAccent) -> Self {
        Self {
            mode,
            accent,
            corner_radius: 6,
        }
    }

    /// Build `egui::Visuals` using [`craft_ui::theme::WidgetPalette`] for consistent controls.
    #[must_use]
    pub fn visuals(self) -> Visuals {
        let dark = self.mode.is_dark();
        let mut visuals = if dark {
            Visuals::dark()
        } else {
            Visuals::light()
        };

        let (palette, fg_color, panel_fill, window_fill, extreme_fill, border_color) = if dark {
            (
                WidgetPalette {
                    noninteractive: Color32::from_rgb(24, 30, 41),
                    inactive: Color32::from_rgb(33, 41, 56),
                    hovered: Color32::from_rgb(45, 56, 76),
                    active: Color32::from_rgb(55, 69, 94),
                    open: Color32::from_rgb(41, 51, 69),
                },
                Color32::from_rgb(226, 232, 240),
                Color32::from_rgb(17, 22, 31),
                Color32::from_rgb(22, 28, 39),
                Color32::from_rgb(13, 17, 24),
                Color32::from_rgb(51, 65, 85),
            )
        } else {
            (
                WidgetPalette {
                    noninteractive: Color32::from_rgb(241, 245, 249),
                    inactive: Color32::from_rgb(226, 232, 240),
                    hovered: Color32::from_rgb(203, 213, 225),
                    active: Color32::from_rgb(186, 200, 218),
                    open: Color32::from_rgb(219, 228, 238),
                },
                Color32::from_rgb(15, 23, 42),
                Color32::from_rgb(248, 250, 252),
                Color32::from_rgb(255, 255, 255),
                Color32::from_rgb(255, 255, 255),
                Color32::from_rgb(203, 213, 225),
            )
        };

        palette.apply_to(
            &mut visuals.widgets,
            CornerRadius::same(self.corner_radius),
            Stroke::new(1.0, fg_color),
        );
        visuals.widgets.noninteractive.bg_stroke = Stroke::new(1.0, border_color);
        visuals.widgets.inactive.bg_stroke = Stroke::new(1.0, border_color);
        visuals.widgets.hovered.bg_stroke = Stroke::new(1.0, self.accent.color32());
        visuals.widgets.active.bg_stroke = Stroke::new(1.2, self.accent.color32());
        visuals.widgets.open.bg_stroke = Stroke::new(1.0, self.accent.color32());

        visuals.panel_fill = panel_fill;
        visuals.window_fill = window_fill;
        visuals.extreme_bg_color = extreme_fill;
        visuals.window_stroke = Stroke::new(1.0, border_color);
        visuals.window_corner_radius = CornerRadius::same(self.corner_radius.saturating_add(2));
        visuals.selection.bg_fill = self.accent.soft_fill(dark);
        visuals.selection.stroke = Stroke::new(1.2, self.accent.color32());
        visuals.hyperlink_color = self.accent.color32();
        visuals
    }

    /// Apply this theme to an `egui::Context`.
    pub fn apply(self, ctx: &egui::Context) {
        ctx.set_visuals(self.visuals());
    }
}
