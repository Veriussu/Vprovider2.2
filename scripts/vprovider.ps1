# ─────────────────────────────────────────────────────────────
#  Bölüm:    Yönetim CLI'si (PowerShell)
#  Dosya:    scripts/vprovider.ps1
#  Amaç:     VProvider yönetim komut tek giriş noktası (Windows)
#  Mekanik:  start|stop|restart|update|clear|delete [all] komutları
#  Kullanım: .\vprovider.ps1 <komut> [seçenekler]
# ─────────────────────────────────────────────────────────────

param(
    [Parameter(Mandatory=$true)]
    [ValidateSet('start','stop','restart','update','clear','delete')]
    [string]$Command,
    [string]$Option
)

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition
$ProjectRoot = Split-Path -Parent $ProjectRoot  # Go up one level from scripts/

function Invoke-Start {
    & "$ProjectRoot\scripts\start.sh" @args
}

function Invoke-Stop {
    & "$ProjectRoot\scripts\stop.sh" @args
}

function Invoke-Restart {
    & "$ProjectRoot\scripts\restart.sh" @args
}

function Invoke-Update {
    Write-Host "🔄 Güncellemeler alınıyor..." -ForegroundColor Yellow
    Set-Location $ProjectRoot
    git fetch origin
    git reset --hard origin/main
    Write-Host "✅ Güncellemeler alındı." -ForegroundColor Green
    Write-Host "📦 Bağımlılıklar güncelleniyor..." -ForegroundColor Yellow
    & "$ProjectRoot\.venv\Scripts\pip.exe" install -r requirements.txt
    & "$ProjectRoot\.venv\Scripts\pip.exe" install -r requirements-hub.txt
    Write-Host "✅ Bağımlılıklar güncellendi." -ForegroundColor Green
}

function Invoke-Clear {
    Write-Host "🧹 Modeller ve veriler temizleniyor..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force "$ProjectRoot\models\*" 2>$null
    Remove-Item -Recurse -Force "$ProjectRoot\data\*" 2>$null
    Remove-Item -Recurse -Force "$ProjectRoot\runtime" 2>$null
    New-Item -ItemType Directory -Force -Path "$ProjectRoot\models" | Out-Null
    New-Item -ItemType Directory -Force -Path "$ProjectRoot\data" | Out-Null
    New-Item -ItemType Directory -Force -Path "$ProjectRoot\runtime" | Out-Null
    New-Item -ItemType File -Force -Path "$ProjectRoot\models\.gitkeep" | Out-Null
    New-Item -ItemType File -Force -Path "$ProjectRoot\data\.gitkeep" | Out-Null
    New-Item -ItemType File -Force -Path "$ProjectRoot\runtime\.gitkeep" | Out-Null
    Write-Host "✅ Modeller ve veriler temizlendi (kod ve .env korunuyor)." -ForegroundColor Green
}

function Invoke-Delete {
    if ($Option -eq 'all') {
        Write-Warning "⚠️  TÜM PROJE SİLİNİYOR! 3 saniye içinde iptal için Ctrl+C..."
        Start-Sleep -Seconds 3
        & "$ProjectRoot\scripts\remove.ps1" --all
    } else {
        & "$ProjectRoot\scripts\remove.ps1"
    }
}

switch ($Command) {
    'start' { Invoke-Start @args }
    'stop' { Invoke-Stop @args }
    'restart' { Invoke-Restart @args }
    'update' { Invoke-Update }
    'clear' { Invoke-Clear }
    'delete' { Invoke-Delete }
    default { 
        Write-Error "Bilinmeyen komut: $Command"
        exit 1
    }
}
