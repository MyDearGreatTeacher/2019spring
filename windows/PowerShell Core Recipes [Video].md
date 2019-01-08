#
```
PowerShell Core Recipes [Video]
TechSnips LLC
Monday, December 31, 2018 
https://www.packtpub.com/virtualization-and-cloud/powershell-core-recipes-video

https://github.com/PacktPublishing/-PowerShell-Core-Recipes
```
### What You Will Learn
```
Install PowerShell Core on the two most common operating systems: Windows and Linux
Collect information from the web by scraping, parsing, or using APIs
Performs tasks by navigating through Windows and Linux file systems and the Windows Registry using PowerShell Core
Remotely manage both Windows and Linux systems with PowerShell Core
Solve common problems by leveraging PowerShell Core 
Deploy and manage common Windows infrastructure services such as DNS, DHCP, and file servers
```
```
INSTALLING POWERSHELL CORE
HARVESTING INFORMATION FROM THE WEB
MANAGING COMMON INFRASTRUCTURE SERVICES
MANAGING REMOTE SYSTEMS
WORKING WITH FILES
WORKING WITH THE WINDOWS REGISTRY
CUSTOMIZING THE ENVIRONMENT
```
# HARVESTING INFORMATION FROM THE WEB

### WebScraping.ps1
```
$Uri = 'https://www.techsnips.io/contributors/'
$Contributors = Invoke-WebRequest -Uri $Uri

$Contributors.Links | select href

$Contributors.Images.where({$_.src -like '*contributors*'}) | foreach {
    $Filename = $_.src.split('/')[-1]
    Invoke-WebRequest -Uri "https://techsnips.io/$($_.src)" -OutFile D:\Images\$Filename
}

Get-ChildItem -Path 'D:\Images'

```

# MANAGING COMMON INFRASTRUCTURE SERVICES

### WorkingwithDNS.ps1
```
# Get this from the PS gallery
# https://www.powershellgallery.com/packages/WindowsCompatibility/1.0.0
Install-Module -Name WindowsCompatibility

Import-WinModule -name dnsserver

# Manager Windows Features
Get-Command -Module dnsserver
(Get-Command -Module dnsserver).Count

# Lets look at some commands
Get-DNSserverZone -ComputerName DC1
Get-DNSServerSetting -Computername DC1

# Lets Pull some records
Get-DnsServerResourceRecord -ComputerName dc1 -ZoneName psdevops.local -RRType A

# Lets Add a primary forward lookup zone
$config = @{
    Name              = 'techsnips.internal'
    ReplicationScope  = 'Forest'
    DynamicUpdate     = 'Secure'
    ComputerName      = 'DC1'
}
Add-DnsServerPrimaryZone @config

Get-DNSserverZone -ComputerName DC1

# Lets Add a primary reverse lookup zone
$config = @{
    Name              = '20.20.10.in-addr.arpa'
    ReplicationScope  = 'Forest'
    DynamicUpdate     = 'Secure'
    ComputerName      = 'DC1'
  }
Add-DnsServerPrimaryZone @config

# Lets test the server
Test-DnsServer -ComputerName DC1 -IPAddress 10.200.0.10 -Zonename techsnips.internal

# Lets Add an A record to our new zone
$arec = @{
    ZoneName      =  'techsnips.internal'
    A              =  $true
    CreatePTR      = $true
    Name           = 'app2'
    AllowUpdateAny =  $true
    IPv4Address    = '10.20.20.1'
    ComputerName      = 'DC1'
}
Add-DnsServerResourceRecord @arec

Get-DnsServerResourceRecord -ComputerName DC1 -ZoneName techsnips.internal -Name 'app2' 

# Lets remove the zone
Remove-DnsServerZone -ComputerName DC1 -name techsnips.internal
Remove-DnsServerZone -ComputerName DC1 -name 20.20.10.in-addr.arpa

# Further commands
Get-command -module dnsserver
```
