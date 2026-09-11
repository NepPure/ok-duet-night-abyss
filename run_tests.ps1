# 运行 tests/ 下所有 test_*.py
# 注意：仓库自带 .venv 优先（见 ok-script/AGENTS.md）；没有则退回全局 python。
$python = "python"
if (Test-Path ".\.venv\Scripts\python.exe") {
    $python = ".\.venv\Scripts\python.exe"
}

Get-ChildItem -Path ".\tests\test_*.py" | ForEach-Object {
    $testFile = $_.FullName
    Write-Host "Running tests in $testFile"
    & $python -m unittest $testFile
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Tests failed in $testFile"
        exit 1
    }
}

Write-Host "All tests passed."
