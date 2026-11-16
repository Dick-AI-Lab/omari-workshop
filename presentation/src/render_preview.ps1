# Render the deck with the locally installed Microsoft PowerPoint (COM):
#   preview/OMARI_From_Pixels_to_Predictions.pdf and preview/slides/slide_NN.png,
# then build preview/contact_sheet.png.  Read-only on the .pptx.
#   powershell -ExecutionPolicy Bypass -File presentation/src/render_preview.ps1
$ErrorActionPreference = 'Stop'
$pres = Split-Path -Parent $PSScriptRoot
$pptx = Join-Path $pres 'OMARI_From_Pixels_to_Predictions.pptx'
$out = Join-Path $pres 'preview'
$slides = Join-Path $out 'slides'
New-Item -ItemType Directory -Force $slides | Out-Null
Get-ChildItem $slides -Filter *.png -ErrorAction SilentlyContinue | Remove-Item -Force
$app = New-Object -ComObject PowerPoint.Application
try {
    $doc = $app.Presentations.Open($pptx, $true, $false, $false)   # ReadOnly, Untitled=false, WithWindow=false
    $doc.SaveAs((Join-Path $out 'OMARI_From_Pixels_to_Predictions.pdf'), 32)  # ppSaveAsPDF
    $i = 0
    foreach ($s in $doc.Slides) {
        $i++
        $s.Export((Join-Path $slides ('slide_{0:D2}.png' -f $i)), 'PNG', 1920, 1080)
    }
    $doc.Close()
} finally {
    $app.Quit()
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($app) | Out-Null
}
& (Join-Path $pres '.venv\Scripts\python.exe') -X utf8 (Join-Path $PSScriptRoot 'contact_sheet.py')
