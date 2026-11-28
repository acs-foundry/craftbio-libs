//! Shared persistent UI preferences backed by [`craft_settings`].

use craft_settings::{LoadReport, Location, Settings, SettingsStore};
use craft_window::policy::TitleBarPref;
use serde::{Deserialize, Serialize};

/// Common desktop UI preferences embedded in each CraftBio application's `Settings` struct.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct CommonUiPrefs {
    pub dark_mode: bool,
    pub use_system_titlebar: bool,
    pub language: String,
    pub font_scale: f32,
}

impl Default for CommonUiPrefs {
    fn default() -> Self {
        Self {
            dark_mode: false,
            use_system_titlebar: false,
            language: "es".to_string(),
            font_scale: 1.0,
        }
    }
}

/// Open a CraftBio application's `config.toml` settings store using [`Location::Auto`].
#[must_use]
pub fn open_settings<T: Settings>() -> (SettingsStore<T>, LoadReport) {
    SettingsStore::<T>::open(Location::Auto)
}

/// Convert the boolean preference into [`craft_window::policy::TitleBarPref`].
#[must_use]
pub const fn titlebar_pref(use_system_titlebar: bool) -> TitleBarPref {
    if use_system_titlebar {
        TitleBarPref::System
    } else {
        TitleBarPref::Integrated
    }
}
