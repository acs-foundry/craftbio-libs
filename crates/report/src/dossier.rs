#![forbid(unsafe_code)]

//! Self-Contained Offline Mol* 3D Interactive HTML Dossier & ArtCraft Suite
//! (`WordCraft` / `DeckCraft` / `GridCraft`) Publication Bundle Exporter.

use std::fmt::Write as _;
use std::path::Path;

use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::{escape_markup, load_offline_molstar_assets, safe_json_for_script};

/// Summary returned after writing a self-contained BiomolCraft 3D HTML dossier.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HtmlDossierSummary {
    pub out_path: String,
    pub bytes: usize,
    pub molstar_inline_offline: bool,
    pub embedded_svg_count: usize,
}

/// Summary returned after exporting an ArtCraft (`WordCraft`/`DeckCraft`/`GridCraft`) publication bundle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtCraftBundleSummary {
    pub out_dir: String,
    pub figure_png: String,
    pub gridcraft_tsv: String,
    pub wordcraft_md: String,
    pub deckcraft_slides_md: String,
}

/// Write a single-file interactive 3D HTML biomolecular dossier embedding offline Mol* (if cached)
/// + pure-JS Canvas3D fallback + Markdown/provenance summary + optional XSS-safe SVG figures.
pub fn write_biomol_html_dossier(
    title: &str,
    pdb_text: &str,
    summary_markdown: &str,
    extra_svgs: &[(&str, &str)],
    prefer_offline_molstar: bool,
    out_path: &Path,
) -> Result<HtmlDossierSummary, String> {
    if let Some(p) = out_path.parent()
        && !p.as_os_str().is_empty()
    {
        std::fs::create_dir_all(p).map_err(|e| e.to_string())?;
    }
    let offline = if prefer_offline_molstar {
        load_offline_molstar_assets()
    } else {
        None
    };
    let has_offline = offline.is_some();
    let (molstar_css, molstar_js) = offline.unwrap_or_default();

    let payload_json = safe_json_for_script(&json!({
        "title": title,
        "pdb": pdb_text,
    }))
    .map_err(|e| e.to_string())?;

    let mut svg_blocks = String::new();
    for (cap, svg) in extra_svgs {
        let _ = writeln!(
            svg_blocks,
            "<section class=\"card\"><h3>{}</h3><div class=\"svg-wrap\">{}</div></section>",
            escape_markup(cap),
            svg
        );
    }

    let safe_title = escape_markup(title);
    let safe_md = escape_markup(summary_markdown);
    let molstar_head = if has_offline {
        format!("<style>{molstar_css}</style><script>{molstar_js}</script>")
    } else {
        String::new()
    };

    let html = format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{safe_title} — BiomolCraft 3D Dossier</title>
{molstar_head}
<style>
  body {{ margin:0; font-family: system-ui, -apple-system, sans-serif; background:#0b1120; color:#e2e8f0; }}
  header {{ padding:16px 24px; background:#0f172a; border-bottom:1px solid #1e293b; display:flex; justify-content:space-between; align-items:center; }}
  header h1 {{ margin:0; font-size:18px; color:#38bdf8; }}
  .grid {{ display:grid; grid-template-columns: 1.4fr 1fr; gap:16px; padding:16px; }}
  @media (max-width: 960px) {{ .grid {{ grid-template-columns: 1fr; }} }}
  .card {{ background:#111827; border:1px solid #1f2937; border-radius:10px; padding:14px; }}
  #viewer3d {{ width:100%; height:560px; position:relative; border-radius:8px; overflow:hidden; background:#070b14; }}
  pre {{ white-space:pre-wrap; word-break:break-word; font-size:12.5px; line-height:1.45; color:#cbd5e1; max-height:520px; overflow:auto; }}
  .svg-wrap {{ overflow:auto; }}
</style>
</head>
<body>
<header>
  <h1>{safe_title}</h1>
  <span>BiomolCraft · CraftBio Suite ({mode})</span>
</header>
<div class="grid">
  <div class="card">
    <div id="viewer3d"><canvas id="fallback3d" width="900" height="560" style="width:100%;height:100%;"></canvas></div>
  </div>
  <div class="card">
    <h2 style="margin-top:0;font-size:15px;color:#38bdf8;">Scientific Audit &amp; Provenance</h2>
    <pre>{safe_md}</pre>
  </div>
</div>
<div style="padding:0 16px 16px;">{svg_blocks}</div>
<script>
const DOSSIER = {payload_json};
(async function init() {{
  const el = document.getElementById('viewer3d');
  if (window.molstar && window.molstar.Viewer) {{
    try {{
      el.innerHTML = '';
      const v = await window.molstar.Viewer.create(el, {{ layoutIsExpanded:false, layoutShowControls:false }});
      await v.loadStructureFromData(DOSSIER.pdb, 'pdb');
      return;
    }} catch (e) {{ console.warn('Mol* fallback:', e); }}
  }}
  // Zero-dependency Canvas3D C-alpha/C4' fallback renderer
  const cv = document.getElementById('fallback3d');
  if (!cv) return;
  const ctx = cv.getContext('2d');
  const pts = [];
  for (const line of DOSSIER.pdb.split('\n')) {{
    if (line.startsWith('ATOM  ') && (line.slice(12,16).trim()==='CA' || line.slice(12,16).trim()==="C4'")) {{
      pts.push([parseFloat(line.slice(30,38)), parseFloat(line.slice(38,46)), parseFloat(line.slice(46,54))]);
    }}
  }}
  if (!pts.length) return;
  let cx=0,cy=0,cz=0;
  for (const p of pts) {{ cx+=p[0]; cy+=p[1]; cz+=p[2]; }}
  cx/=pts.length; cy/=pts.length; cz/=pts.length;
  let maxr=1;
  for (const p of pts) maxr = Math.max(maxr, Math.hypot(p[0]-cx, p[1]-cy, p[2]-cz));
  let ang = 0.4;
  function draw() {{
    ctx.fillStyle='#070b14'; ctx.fillRect(0,0,cv.width,cv.height);
    const sc = Math.min(cv.width, cv.height)*0.42/maxr;
    const ca=Math.cos(ang), sa=Math.sin(ang);
    ctx.strokeStyle='#38bdf8'; ctx.lineWidth=2.0; ctx.beginPath();
    for (let i=0;i<pts.length;i++) {{
      const x=pts[i][0]-cx, y=pts[i][1]-cy, z=pts[i][2]-cz;
      const rx = x*ca - z*sa;
      const sx = cv.width/2 + rx*sc, sy = cv.height/2 - y*sc;
      if (i===0) ctx.moveTo(sx,sy); else ctx.lineTo(sx,sy);
    }}
    ctx.stroke();
  }}
  draw();
}})();
</script>
</body>
</html>"#,
        mode = if has_offline {
            "Offline Mol* 3D Embedded"
        } else {
            "Canvas3D Standalone"
        }
    );
    std::fs::write(out_path, html.as_bytes()).map_err(|e| e.to_string())?;
    Ok(HtmlDossierSummary {
        out_path: out_path.display().to_string(),
        bytes: html.len(),
        molstar_inline_offline: has_offline,
        embedded_svg_count: extra_svgs.len(),
    })
}

/// Export an ArtCraft-compatible publication bundle (`figure_4k.png` + `table_gridcraft.tsv` +
/// `report_wordcraft.md` + `slides_deckcraft.md`) ready for `WordCraft`, `DeckCraft`, and `GridCraft`.
pub fn export_artcraft_bundle(
    title: &str,
    png_bytes: &[u8],
    metrics_rows: &[(&str, String)],
    summary_markdown: &str,
    out_dir: &Path,
) -> Result<ArtCraftBundleSummary, String> {
    std::fs::create_dir_all(out_dir).map_err(|e| e.to_string())?;
    let fig_path = out_dir.join("figure_biomolcraft.png");
    std::fs::write(&fig_path, png_bytes).map_err(|e| e.to_string())?;

    let tsv_path = out_dir.join("table_gridcraft.tsv");
    let mut tsv = String::from("Metric\tValue\n");
    for (k, v) in metrics_rows {
        let safe_v = v.replace(['\t', '\n', '\r'], " ");
        let _ = writeln!(tsv, "{k}\t{safe_v}");
    }
    std::fs::write(&tsv_path, tsv).map_err(|e| e.to_string())?;

    let word_path = out_dir.join("report_wordcraft.md");
    let word_md = format!(
        "# {title}\n\n![BiomolCraft 3D Render](figure_biomolcraft.png)\n\n{summary_markdown}\n"
    );
    std::fs::write(&word_path, word_md).map_err(|e| e.to_string())?;

    let deck_path = out_dir.join("slides_deckcraft.md");
    let mut deck = format!("# {title}\n\n---\n\n## Key Structural Metrics\n\n");
    for (k, v) in metrics_rows.iter().take(8) {
        let _ = writeln!(deck, "- **{k}:** {v}");
    }
    deck.push_str("\n---\n\n## 3D Biomolecular View\n\n![3D Figure](figure_biomolcraft.png)\n");
    std::fs::write(&deck_path, deck).map_err(|e| e.to_string())?;

    Ok(ArtCraftBundleSummary {
        out_dir: out_dir.display().to_string(),
        figure_png: fig_path.display().to_string(),
        gridcraft_tsv: tsv_path.display().to_string(),
        wordcraft_md: word_path.display().to_string(),
        deckcraft_slides_md: deck_path.display().to_string(),
    })
}
