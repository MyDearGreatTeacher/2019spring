
```
Mastering Python Forensics
Dr. Michael Spreitzenbarth, Dr. Johann Uhrmann
October 2015
```

What You Will Learn
```
Explore the forensic analysis of different platforms such as Windows, Android, and vSphere
Semi-automatically reconstruct major parts of the system activity and time-line
Leverage Python ctypes for protocol decoding
Examine artifacts from mobile, Skype, and browsers
Discover how to utilize Python to improve the focus of your analysis
Investigate in volatile memory with the help of volatility on the Android and Linux platforms
```

```
1: SETTING UP THE LAB AND INTRODUCTION TO PYTHON CTYPES
Setting up the Lab
Introduction to Python ctypes

2: FORENSIC ALGORITHMS
Algorithms
Supporting the chain of custody
Real-world scenarios

3: USING PYTHON FOR WINDOWS AND LINUX FORENSICS
Analyzing the Windows Event Log
Analyzing the Windows Registry
Implementing Linux specific checks

4: USING PYTHON FOR NETWORK FORENSICS
Using Dshell during an investigation
Using Scapy during an investigation

5: USING PYTHON FOR VIRTUALIZATION FORENSICS
Considering virtualization as a new attack surface
Searching for misuse of virtual resources
Using virtualization as a source of evidence

6: USING PYTHON FOR MOBILE FORENSICS
The investigative model for smartphones
Android
Apple iOS

7: USING PYTHON FOR MEMORY FORENSICS
Understanding Volatility basics
Using Volatility on Android
Using Volatility on Linux

```
# 1
```
from ctypes import *

class case(Union):
	_fields_ = [
	("evidence_long", c_long),
	("evidence_int", c_int),
	("evidence_char", c_char * 4),
	]

value = raw_input("Enter new evidence number:")
new_evidence = case(int(value))
print "Evidence number as a long: %ld" % new_evidence.evidence_long
print "Evidence number as a int: %d" % new_evidence.evidence_int
print "Evidence number as a char: %s" % new_evidence.evidence_char

```
# 2
```
#!/usr/bin/python

import socket

NSRL_SERVER='127.0.0.1'
NSRL_PORT=9120

def nsrlquery(md5hashes):
    """Query the NSRL server and return a list of booleans.

    Arguments:
    md5hashes -- The list of MD5 hashes for the query.
    """

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((NSRL_SERVER, NSRL_PORT))

    try:
        f = s.makefile('r')
        s.sendall("version: 2.0\r\n")
        response = f.readline();
        if response.strip() != 'OK':
            raise RuntimeError('NSRL handshake error')

        query = 'query ' + ' '.join(md5hashes) + "\r\n"
        s.sendall(query)
        response = f.readline();

        if response[:2] != 'OK':
            raise RuntimeError('NSRL query error')

        return [c=='1' for c in response[3:].strip()]
    finally:
        s.close()
```

```
#!/usr/bin/python

import hashlib
import sys

def multi_hash(filename):
    """Calculates the md5 and sha256 hashes
       of the specified file and returns a list
       containing the hash sums as hex strings."""

    md5 = hashlib.md5()
    sha256 = hashlib.sha256()

    with open(filename, 'rb') as f:
        while True:
            buf = f.read(2**20)
            if not buf:
                break
            md5.update(buf)
            sha256.update(buf)

    return [md5.hexdigest(), sha256.hexdigest()]


if __name__ == '__main__':
    hashes = []
    print '---------- MD5 sums ----------'
    for filename in sys.argv[1:]:
        h = multi_hash(filename)
        hashes.append(h)
        print '%s  %s' % (h[0], filename)
        
    print '---------- SHA256 sums ----------'
    for i in range(len(hashes)):
        print '%s  %s' % (hashes[i][1], sys.argv[i+1])
```
