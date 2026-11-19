$path = 'F:\ai\rfdetr\model.safetensors'
$fs = [IO.File]::OpenRead($path)
$br = New-Object IO.BinaryReader($fs)
$len = $br.ReadInt64()
$buf = $br.ReadBytes($len)
$json = [Text.Encoding]::UTF8.GetString($buf)
$fs.Close()

# 键名形如 "name":{...}，用正则取出所有键
$keys = [regex]::Matches($json, '"([A-Za-z0-9_.]+)":\{') | ForEach-Object { $_.Groups[1].Value }
Write-Output ("total keys: " + $keys.Count)
$keys | Where-Object { $_ -match 'ref_point|projector|query|refpoint|class_embed|label' }
