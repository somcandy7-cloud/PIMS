# Design System Document

## 1. Overview & Creative North Star: "The Industrial Architect"

This design system transcends the standard "corporate dashboard" by adopting the persona of **The Industrial Architect**. It reflects the core of the POSCO identity—strength, precision, and structural integrity—while layering in the sophistication of high-end AI analysis. 

Instead of a flat, grid-based layout, we treat the UI as an architectural environment. We use **Intentional Asymmetry** to guide the eye toward critical anomalies and **Tonal Depth** to signify data importance. The experience is not just "clean"; it is authoritative. We replace the clutter of traditional borders with breathing room and sophisticated material layers, making the complex data of AI anomaly detection feel like an editorial spread in a premium engineering journal.

---

## 2. Colors: Tonal Integrity & The "No-Line" Rule

The palette is rooted in the deep, commanding `primary` (POSCO BLUE) and the technical clarity of `secondary` (POSCO LIGHT BLUE). To achieve a premium feel, we strictly follow these structural rules:

*   **The "No-Line" Rule:** 1px solid borders are strictly prohibited for defining sections. Boundaries must be established through **background color shifts**. For example, a main content area using `surface` will host modules set in `surface_container_low`. This creates a seamless, modern flow that feels engineered rather than "boxed in."
*   **Surface Hierarchy & Nesting:** Use the surface-container tiers to create physical depth. 
    *   **Base Layer:** `surface` (Background).
    *   **Sectional Layers:** `surface_container_low` for broad groupings.
    *   **Interactive/Elevated Modules:** `surface_container_lowest` for cards that need to "pop" against a low-tier background.
*   **The Glass & Gradient Rule:** To signify the "AI" layer of the application, use **Glassmorphism** for floating sidebars or modal overlays. Utilize `surface_container_highest` with a 60% opacity and a 16px backdrop-blur. 
*   **Signature Textures:** For primary action buttons or high-level anomaly alerts, use a subtle linear gradient transitioning from `primary` (#00385b) to `primary_container` (#05507d). This provides a "machined metal" finish that flat color cannot replicate.

---

## 3. Typography: The Balance of Strength and Precision

We utilize two distinct typefaces to create a high-contrast editorial hierarchy:
*   **Manrope (Display & Headlines):** Used for large metrics and section headers. Its geometric construction mirrors the "Industrial Architect" aesthetic—robust and modern.
*   **Inter (Body & Labels):** Used for data tables, chat interfaces, and technical labels. It offers unparalleled legibility for dense AI-generated insights.

**Key Scales:**
*   **Display-LG (Manrope):** For critical anomaly scores. Use a tight letter-spacing (-0.02em) to feel intentional and authoritative.
*   **Headline-SM (Manrope):** For module titles. This should always be paired with `on_surface` to maintain high visual weight.
*   **Label-MD (Inter):** For data table headers and slider labels. Use `outline` color to de-emphasize secondary information while maintaining clarity.

---

## 4. Elevation & Depth: Tonal Layering

Depth in this system is achieved through **Tonal Layering** rather than traditional drop shadows.

*   **The Layering Principle:** Stack `surface_container_lowest` modules on top of `surface_container_low` sections. The slight shift in luminosity provides a sophisticated "lift."
*   **Ambient Shadows:** For floating elements (like the AI Chat interface), use a highly diffused shadow: `y: 8px, blur: 24px, color: rgba(23, 29, 29, 0.06)`. By tinting the shadow with the `on_surface` color, the element feels part of the atmosphere rather than a separate object.
*   **The Ghost Border:** If a boundary is required for accessibility in data-heavy views, use a "Ghost Border"—the `outline_variant` token at **15% opacity**. It should be felt, not seen.

---

## 5. Components: Bespoke Implementation

### 5.1. Sidebar & Navigation
*   **Style:** Fixed to the left with a `surface_container_low` background. 
*   **File Uploader:** Do not use a dashed box. Use a `surface_container_high` area with a `secondary` icon. The "drop zone" should feel like a solid, receptive technical slot.
*   **Sliders:** Use `primary` for the track and `primary_fixed` for the handle. The handle should have a subtle `ambient shadow` to appear tactile.

### 5.2. Data Tables & Lists
*   **Rule:** No horizontal or vertical dividers. 
*   **Implementation:** Use a 4px vertical gap between rows. The row background should be `surface_container_lowest`. When hovered, the row shifts to `primary_fixed` at 20% opacity.
*   **Typography:** Column headers use `label-md` in all-caps with 0.05em tracking for a professional, technical look.

### 5.3. AI Chat Interface
*   **Structure:** A floating glass panel (`surface_container_highest` + blur). 
*   **Messages:** User messages in `primary_container`; AI responses in `surface_container_low`. Use `rounded-lg` (0.5rem) for corners to maintain a "clean-industrial" feel.

### 5.4. Image Viewer & Indicators
*   **Viewer:** Deep `on_surface` (Black) background to let the imagery (industrial parts/scans) stand out.
*   **Indicators:** Anomaly markers should use the `error` token with a pulsing `error_container` glow. Use `secondary_fixed` for "safe" AI-detected points of interest.

### 5.5. Charts (Bar & Line)
*   **Visual Soul:** Line charts should use a `secondary` stroke with a soft gradient fill beneath the line, fading from `secondary_container` to transparent. 
*   **Minimalism:** Remove all grid lines except for the baseline. Use `label-sm` for axes.

---

## 6. Do's and Don'ts

### Do:
*   **Do** use vertical whitespace to separate modules instead of lines.
*   **Do** use the `tertiary` tokens (warm ambers) sparingly for "Warning" states to contrast against the cool POSCO blues.
*   **Do** ensure that the AI "insights" are always highlighted using the `primary_fixed` background for high-priority callouts.

### Don't:
*   **Don't** use 100% black for text. Use `on_surface` (#171d1d) to maintain a premium, softer contrast.
*   **Don't** use "default" system shadows. They look cheap and break the "Architect" aesthetic.
*   **Don't** overcrowd the sidebar. If you have more than 5 sliders, group them into collapsible `surface_container_high` accordions.
*   **Don't** use sharp 0px corners. Stick to the `md` (0.375rem) or `lg` (0.5rem) roundedness to keep the engineering tool feeling modern and ergonomic.