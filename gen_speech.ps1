Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.SetOutputToWaveFile('speech_en.wav')
$synth.Speak('What is the character of the university school?')
$synth.Dispose()
Write-Host "Audio generated successfully"
