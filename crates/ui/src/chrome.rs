//! Reusable CraftBio scientific chrome widgets for `egui 0.37`.

use egui::{Color32, CornerRadius, Frame, Margin, RichText, Stroke, Ui};

use crate::theme::ThemeMode;

/// Compact semantic pill badge (e.g., TPP pass/warn/fail, hERG risk, AD status).
pub fn badge(ui: &mut Ui, label: &str, color: Color32) {
    let bg = Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), 36);
    Frame::NONE
        .fill(bg)
        .stroke(Stroke::new(1.0, color))
        .corner_radius(CornerRadius::same(10))
        .inner_margin(Margin::symmetric(8, 2))
        .show(ui, |ui| {
            ui.label(RichText::new(label).small().strong().color(color));
        });
}

/// Rounded scientific card container with optional accent border.
pub fn section_card<R>(
    ui: &mut Ui,
    accent: Option<Color32>,
    add_contents: impl FnOnce(&mut Ui) -> R,
) -> egui::InnerResponse<R> {
    let mut frame = Frame::group(ui.style())
        .corner_radius(CornerRadius::same(8))
        .inner_margin(Margin::same(10));
    if let Some(color) = accent {
        frame = frame.stroke(Stroke::new(1.2, color));
    }
    frame.show(ui, add_contents)
}

/// Striped two-column scientific key-value metrics grid.
pub fn kv_grid(ui: &mut Ui, id: &str, rows: &[(&str, String)]) {
    egui::Grid::new(id)
        .striped(true)
        .spacing([20.0, 4.0])
        .show(ui, |ui| {
            for (k, v) in rows {
                ui.label(*k);
                ui.strong(v);
                ui.end_row();
            }
        });
}

/// Compact button to toggle light/dark mode; returns `true` when toggled.
pub fn theme_toggle_button(ui: &mut Ui, mode: &mut ThemeMode) -> bool {
    if ui.small_button(mode.switch_label()).clicked() {
        mode.toggle();
        true
    } else {
        false
    }
}

/// Standard CraftBio bottom status bar using `egui 0.37`'s [`egui::Panel::bottom`].
pub fn status_bar(ui: &mut Ui, id: &str, left: impl FnOnce(&mut Ui), right: impl FnOnce(&mut Ui)) {
    egui::Panel::bottom(id).show(ui, |ui| {
        ui.horizontal(|ui| {
            left(ui);
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                right(ui);
            });
        });
    });
}
