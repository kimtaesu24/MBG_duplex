#!/usr/bin/env python3

from __future__ import annotations

import html
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PPTX = ROOT / "softvq_main_figure_editable.pptx"

SLIDE_W = 13.333
SLIDE_H = 7.5
EMU = 914400


def emu(v: float) -> int:
    return int(round(v * EMU))


def color(hex_color: str) -> str:
    return hex_color.strip("#").upper()


def write(zf: zipfile.ZipFile, name: str, text: str) -> None:
    zf.writestr(name, text.encode("utf-8"))


class Slide:
    def __init__(self) -> None:
        self.items: list[str] = []
        self.shape_id = 2

    def next_id(self) -> int:
        sid = self.shape_id
        self.shape_id += 1
        return sid

    def rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        text: str = "",
        fill: str = "FFFFFF",
        line: str = "263238",
        font: int = 10,
        bold: bool = False,
        align: str = "ctr",
        valign: str = "mid",
        radius: bool = True,
        name: str = "shape",
    ) -> None:
        sid = self.next_id()
        prst = "roundRect" if radius else "rect"
        paras = []
        for i, line_text in enumerate(text.split("\n")):
            esc = html.escape(line_text)
            btag = ' b="1"' if bold or i == 0 and font >= 13 else ""
            paras.append(
                f"""<a:p><a:pPr algn="{align}"/><a:r><a:rPr lang="en-US" sz="{font*100}"{btag}><a:solidFill><a:srgbClr val="111827"/></a:solidFill></a:rPr><a:t>{esc}</a:t></a:r></a:p>"""
            )
        body = "".join(paras) if paras else "<a:p/>"
        self.items.append(
            f"""<p:sp>
  <p:nvSpPr><p:cNvPr id="{sid}" name="{html.escape(name)}"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>
  <p:spPr>
    <a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/><a:ext cx="{emu(w)}" cy="{emu(h)}"/></a:xfrm>
    <a:prstGeom prst="{prst}"><a:avLst/></a:prstGeom>
    <a:solidFill><a:srgbClr val="{color(fill)}"/></a:solidFill>
    <a:ln w="12700"><a:solidFill><a:srgbClr val="{color(line)}"/></a:solidFill></a:ln>
  </p:spPr>
  <p:txBody><a:bodyPr wrap="square" anchor="{valign}"/><a:lstStyle/>{body}</p:txBody>
</p:sp>"""
        )

    def text(self, x: float, y: float, w: float, h: float, text: str, font: int = 12, fill: str = "111827", bold: bool = False, align: str = "ctr") -> None:
        sid = self.next_id()
        paras = []
        for line_text in text.split("\n"):
            esc = html.escape(line_text)
            btag = ' b="1"' if bold else ""
            paras.append(
                f"""<a:p><a:pPr algn="{align}"/><a:r><a:rPr lang="en-US" sz="{font*100}"{btag}><a:solidFill><a:srgbClr val="{color(fill)}"/></a:solidFill></a:rPr><a:t>{esc}</a:t></a:r></a:p>"""
            )
        self.items.append(
            f"""<p:sp>
  <p:nvSpPr><p:cNvPr id="{sid}" name="text"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>
  <p:spPr><a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/><a:ext cx="{emu(w)}" cy="{emu(h)}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/><a:ln><a:noFill/></a:ln></p:spPr>
  <p:txBody><a:bodyPr wrap="square"/><a:lstStyle/>{"".join(paras)}</p:txBody>
</p:sp>"""
        )

    def line(self, x1: float, y1: float, x2: float, y2: float, line: str = "1F2933", dashed: bool = False, width: int = 2, arrow: bool = True) -> None:
        sid = self.next_id()
        x = min(x1, x2)
        y = min(y1, y2)
        w = abs(x2 - x1) or 0.01
        h = abs(y2 - y1) or 0.01
        flip_h = ' flipH="1"' if x2 < x1 else ""
        flip_v = ' flipV="1"' if y2 < y1 else ""
        dash = '<a:prstDash val="dash"/>' if dashed else ""
        head = '<a:headEnd type="triangle"/>' if arrow else ""
        self.items.append(
            f"""<p:cxnSp>
  <p:nvCxnSpPr><p:cNvPr id="{sid}" name="arrow"/><p:cNvCxnSpPr/><p:nvPr/></p:nvCxnSpPr>
  <p:spPr>
    <a:xfrm{flip_h}{flip_v}><a:off x="{emu(x)}" y="{emu(y)}"/><a:ext cx="{emu(w)}" cy="{emu(h)}"/></a:xfrm>
    <a:prstGeom prst="line"><a:avLst/></a:prstGeom>
    <a:ln w="{width*12700}"><a:solidFill><a:srgbClr val="{color(line)}"/></a:solidFill>{dash}{head}</a:ln>
  </p:spPr>
</p:cxnSp>"""
        )

    def add_plus(self, x: float, y: float) -> None:
        self.rect(x - 0.12, y - 0.12, 0.24, 0.24, "+", "FFFFFF", "1F2933", font=14, bold=True, name="add")

    def add_times(self, x: float, y: float) -> None:
        self.rect(x - 0.12, y - 0.12, 0.24, 0.24, "×", "FFFFFF", "1F2933", font=14, bold=True, name="multiply")


def build_slide() -> Slide:
    s = Slide()
    s.rect(0.08, 0.08, 1.36, 6.62, "", "FBF7FF", "6F4AA0", radius=True, name="Inputs panel")
    s.text(0.33, 0.14, 0.9, 0.25, "1. INPUTS", 14, "6F4AA0", True)
    s.rect(0.15, 0.55, 1.22, 1.15, "Streaming Mimi\nAudio Feature\n\n[B,T_audio,512]", "FFFFFF", "6F4AA0", 9)
    s.rect(0.15, 1.93, 1.22, 1.15, "LLM Feature\n\n[B,T_audio,4096]\ncurrently zero dummy", "FFFFFF", "6F4AA0", 9)
    s.rect(0.15, 3.38, 1.22, 0.9, "External inputs only\n\nNo ground-truth motion\nis fed as an input", "FFFFFF", "6F4AA0", 8)

    s.rect(1.75, 0.08, 5.75, 2.78, "", "F4F9FF", "1D5FA8", name="Audio panel")
    s.text(3.75, 0.14, 1.8, 0.25, "2. AUDIO STREAM", 14, "1D5FA8", True)
    s.rect(1.95, 0.78, 1.0, 0.48, "audio_feat\n[B,T_audio,512]", "FFFFFF", "1D5FA8", 8)
    s.text(3.05, 0.75, 0.55, 0.45, "temporal\nupsample ×2", 7, "111827")
    s.rect(3.68, 0.75, 0.78, 0.55, "audio_proj\n512→H", "F0F7FF", "1D5FA8", 8)
    s.rect(1.95, 1.72, 1.0, 0.48, "llm_feat\n[B,T_audio,4096]", "FFFFFF", "1D5FA8", 8)
    s.text(3.05, 1.69, 0.55, 0.45, "temporal\nupsample ×2", 7, "111827")
    s.rect(3.68, 1.69, 0.78, 0.55, "llm_proj\n4096→H", "F0F7FF", "1D5FA8", 8)
    s.rect(4.85, 0.72, 0.9, 0.72, "learned\npos + phase\n[B,T_face,H]", "FFFFFF", "1D5FA8", 7)
    s.add_plus(4.95, 1.68)
    s.rect(5.42, 1.18, 1.45, 0.86, "Causal Transformer Encoder\n(audio_encoder)\nL=6, heads=8", "EAF4FF", "1D5FA8", 8)
    s.rect(7.05, 1.28, 0.34, 0.66, "Layer\nNorm", "EAF4FF", "1D5FA8", 7)
    s.rect(3.96, 2.33, 0.9, 0.42, "z_head\nH→code_dim", "F2FBEE", "3C873F", 7)
    s.rect(5.25, 2.33, 1.08, 0.42, "prior_head\n(H+code_dim)→54", "FFF3E8", "DD7D2D", 7)

    s.line(1.37, 1.12, 1.95, 1.02)
    s.line(1.37, 2.5, 1.95, 1.96)
    s.line(2.95, 1.02, 3.68, 1.02)
    s.line(2.95, 1.96, 3.68, 1.96)
    s.line(4.46, 1.02, 4.83, 1.55)
    s.line(4.46, 1.96, 4.83, 1.78)
    s.line(5.18, 1.68, 5.42, 1.61)
    s.line(5.75, 1.44, 4.95, 1.56, dashed=True)
    s.line(6.87, 1.61, 7.05, 1.61)
    s.line(7.22, 1.94, 5.79, 2.33, dashed=True)
    s.line(7.22, 1.94, 4.4, 2.33, dashed=True)

    s.rect(7.58, 0.08, 5.55, 2.78, "", "F5FFF3", "3C873F", name="Motion panel")
    s.text(9.25, 0.14, 1.8, 0.25, "3. MOTION STREAM", 14, "3C873F", True)
    s.rect(7.78, 1.05, 0.95, 1.05, "Motion State\ntrain: shifted GT\ninfer: pred\n[B,T_face,54]", "F7F1FF", "7D59B0", 7)
    s.rect(7.78, 2.18, 0.95, 0.44, "Audio Hidden\n[B,T_face,H]", "F0F7FF", "1D5FA8", 7)
    s.rect(9.03, 1.45, 0.92, 0.55, "motion_proj\n(54+H)→H", "F2FBEE", "3C873F", 8)
    s.rect(10.14, 0.6, 0.86, 0.62, "learned\npos + phase", "FFFFFF", "3C873F", 7)
    s.add_plus(10.33, 1.72)
    s.rect(10.78, 1.15, 1.5, 0.9, "Causal Transformer Encoder\n(motion_encoder)\nL=6, heads=8", "F2FBEE", "3C873F", 8)
    s.rect(12.48, 1.35, 0.38, 0.58, "Layer\nNorm", "F2FBEE", "3C873F", 7)
    s.line(7.5, 1.61, 7.78, 1.48)
    s.line(7.5, 1.61, 7.78, 2.4)
    s.line(8.73, 1.55, 9.03, 1.72)
    s.line(9.95, 1.72, 10.2, 1.72)
    s.line(10.33, 1.22, 10.33, 1.6)
    s.line(10.45, 1.72, 10.78, 1.6)
    s.line(12.28, 1.6, 12.48, 1.64)

    s.rect(2.95, 3.15, 7.78, 3.28, "", "FFF9F2", "DD7D2D", name="Dynamics panel")
    s.text(5.35, 3.2, 2.4, 0.25, "4. DYNAMICS + FUSION", 14, "DD7D2D", True)
    s.rect(3.18, 3.72, 0.9, 0.44, "motion_h\n[B,T,H]", "F2FBEE", "3C873F", 7)
    s.rect(3.18, 4.28, 0.9, 0.44, "audio_h\n[B,T,H]", "F0F7FF", "1D5FA8", 7)
    s.add_plus(4.38, 4.0)
    s.rect(4.75, 3.62, 0.9, 0.48, "delta_head\n2H→54", "FFF3E8", "DD7D2D", 8)
    s.rect(4.75, 4.28, 0.9, 0.48, "residual_head\n2H→54", "FFF0F0", "BD4A4A", 8)
    s.rect(5.95, 3.62, 0.85, 0.48, "scale/tanh\nexpr .08\njaw .03", "FFF0F0", "BD4A4A", 7)
    s.rect(5.95, 4.28, 0.85, 0.48, "scale/tanh\nexpr .08\njaw .03", "FFF0F0", "BD4A4A", 7)
    s.rect(7.4, 3.65, 1.14, 0.48, "candidate_motion\nprev + delta", "F7F1FF", "7D59B0", 8)
    s.rect(4.55, 5.02, 1.25, 0.76, "gate inputs\n[dyn_h, prev, prior,\ncandidate, cand-prior]", "F6F7F8", "7B8490", 7)
    s.rect(6.15, 5.12, 1.05, 0.54, "gate_head\n(2H+4×54)→3", "FFF3E8", "DD7D2D", 8)
    s.rect(7.48, 5.0, 0.92, 0.75, "sigmoid\nexpr / jaw / neck", "F7F1FF", "7D59B0", 8)
    s.rect(8.62, 5.0, 1.08, 0.75, "expand to 54D\n+ warmup gate", "F6F7F8", "7B8490", 8)
    s.add_times(9.95, 4.82)
    s.rect(8.62, 5.95, 1.25, 0.36, "base = gate*candidate\n+ (1-gate)*prior", "FFFFFF", "DD7D2D", 7)
    s.add_plus(10.47, 4.33)
    s.text(6.9, 6.15, 2.5, 0.2, "pred = base + residual", 9, "111827", True)
    s.line(4.08, 3.94, 4.26, 3.98)
    s.line(4.08, 4.5, 4.26, 4.06)
    s.line(4.5, 4.0, 4.75, 3.86)
    s.line(4.5, 4.0, 4.75, 4.52)
    s.line(5.65, 3.86, 5.95, 3.86)
    s.line(5.65, 4.52, 5.95, 4.52)
    s.line(6.8, 3.86, 7.4, 3.9)
    s.line(8.54, 3.9, 9.83, 4.72)
    s.line(5.8, 5.4, 6.15, 5.39)
    s.line(7.2, 5.39, 7.48, 5.38)
    s.line(8.4, 5.38, 8.62, 5.38)
    s.line(9.7, 5.38, 9.86, 4.93)
    s.line(10.07, 4.82, 10.35, 4.42)
    s.line(6.8, 4.52, 10.35, 4.28)
    s.line(9.25, 6.0, 10.36, 4.42)

    s.rect(10.98, 3.15, 2.22, 1.65, "", "FBF8FF", "7D59B0", name="Output panel")
    s.text(11.55, 3.22, 0.9, 0.25, "5. OUTPUT", 14, "7D59B0", True)
    s.rect(11.18, 3.6, 1.82, 0.82, "pred_motion\n54D continuous FLAME motion\n[B,T_face,54]", "FFFFFF", "7D59B0", 8)
    s.line(10.59, 4.33, 11.18, 4.01)

    s.rect(10.98, 4.98, 2.22, 1.45, "", "FBF8FF", "7D59B0", name="Loss panel")
    s.text(11.33, 5.04, 1.35, 0.25, "6. TRAINING LOSSES", 13, "7D59B0", True)
    s.rect(11.17, 5.38, 1.82, 0.42, "z loss vs codec quant_to_sum_feat(gt)", "F2FBEE", "3C873F", 7)
    s.rect(11.17, 5.92, 1.82, 0.42, "motion / jaw / vel losses vs GT", "FFF0F0", "BD4A4A", 7)
    s.line(4.42, 2.75, 11.17, 5.59, dashed=True)
    s.line(11.75, 4.42, 11.75, 5.92, dashed=True)
    s.line(11.75, 4.42, 7.98, 1.05, dashed=True)
    s.line(11.75, 4.42, 7.98, 1.05, dashed=True)

    s.rect(0.08, 6.88, 13.05, 0.52, "Legend / notes: all encoders are causal. Motion state is internal: shifted GT during training, previous prediction during inference. Output is continuous 54D FLAME motion.", "FFFFFF", "CBD5E1", 8, align="l", radius=True)
    return s


def make_pptx(slide: Slide) -> None:
    slide_cx = emu(SLIDE_W)
    slide_cy = emu(SLIDE_H)
    sp_tree = "\n".join(slide.items)
    with zipfile.ZipFile(PPTX, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        write(zf, "[Content_Types].xml", """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>
<Override PartName="/ppt/slides/slide1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>
<Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>
<Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>
<Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>
<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>""")
        write(zf, "_rels/.rels", """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/>
<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>""")
        write(zf, "docProps/core.xml", """<?xml version="1.0" encoding="UTF-8" standalone="yes"?><cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>SoftVQ Editable Figure</dc:title><dc:creator>Codex</dc:creator></cp:coreProperties>""")
        write(zf, "docProps/app.xml", """<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"><Application>Codex</Application><Slides>1</Slides></Properties>""")
        write(zf, "ppt/presentation.xml", f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
<p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId1"/></p:sldMasterIdLst><p:sldIdLst><p:sldId id="256" r:id="rId2"/></p:sldIdLst><p:sldSz cx="{slide_cx}" cy="{slide_cy}" type="screen16x9"/><p:notesSz cx="6858000" cy="9144000"/></p:presentation>""")
        write(zf, "ppt/_rels/presentation.xml.rels", """<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="slideMasters/slideMaster1.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide1.xml"/><Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="theme/theme1.xml"/></Relationships>""")
        write(zf, "ppt/slides/slide1.xml", f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"><p:cSld><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>{sp_tree}</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sld>""")
        write(zf, "ppt/slides/_rels/slide1.xml.rels", """<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/></Relationships>""")
        write(zf, "ppt/slideMasters/slideMaster1.xml", """<?xml version="1.0" encoding="UTF-8" standalone="yes"?><p:sldMaster xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"><p:cSld><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr><p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst></p:sldMaster>""")
        write(zf, "ppt/slideMasters/_rels/slideMaster1.xml.rels", """<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="../theme/theme1.xml"/></Relationships>""")
        write(zf, "ppt/slideLayouts/slideLayout1.xml", """<?xml version="1.0" encoding="UTF-8" standalone="yes"?><p:sldLayout xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" type="blank" preserve="1"><p:cSld name="Blank"><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sldLayout>""")
        write(zf, "ppt/slideLayouts/_rels/slideLayout1.xml.rels", """<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"/>""")
        write(zf, "ppt/theme/theme1.xml", """<?xml version="1.0" encoding="UTF-8" standalone="yes"?><a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="Office"><a:themeElements><a:clrScheme name="Office"><a:dk1><a:srgbClr val="000000"/></a:dk1><a:lt1><a:srgbClr val="FFFFFF"/></a:lt1><a:dk2><a:srgbClr val="1F2933"/></a:dk2><a:lt2><a:srgbClr val="EEECE1"/></a:lt2><a:accent1><a:srgbClr val="4472C4"/></a:accent1><a:accent2><a:srgbClr val="70AD47"/></a:accent2><a:accent3><a:srgbClr val="FFC000"/></a:accent3><a:accent4><a:srgbClr val="5B9BD5"/></a:accent4><a:accent5><a:srgbClr val="A5A5A5"/></a:accent5><a:accent6><a:srgbClr val="ED7D31"/></a:accent6><a:hlink><a:srgbClr val="0563C1"/></a:hlink><a:folHlink><a:srgbClr val="954F72"/></a:folHlink></a:clrScheme><a:fontScheme name="Office"><a:majorFont><a:latin typeface="Arial"/></a:majorFont><a:minorFont><a:latin typeface="Arial"/></a:minorFont></a:fontScheme><a:fmtScheme name="Office"><a:fillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:fillStyleLst><a:lnStyleLst><a:ln w="6350"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln></a:lnStyleLst><a:effectStyleLst><a:effectStyle><a:effectLst/></a:effectStyle></a:effectStyleLst><a:bgFillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:bgFillStyleLst></a:fmtScheme></a:themeElements></a:theme>""")


def main() -> None:
    make_pptx(build_slide())
    print(PPTX)


if __name__ == "__main__":
    main()
