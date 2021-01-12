'''Get host/gpu specs.'''

import re
import socket
import subprocess

from dataclasses import dataclass
from pathlib import Path as path
from textwrap import dedent

@dataclass
class MachineSpecs:
    hostname: str
    cpu: str
    kernel: str
    ram: str
    distro: str
    rocmversion: str
    vbios: str
    gpuid: str
    deviceinfo: str
    vram: str
    perflevel: str
    mclk: str
    sclk: str
    bandwidth: str

    def __str__(self):
        return  dedent(f'''\
        Host info:
            hostname:       {self.hostname}
            cpu info:       {self.cpu}
            ram:            {self.ram}
            distro:         {self.distro}
            kernel version: {self.kernel}
            rocm version:   {self.rocmversion}
        Device info:
            device:            {self.deviceinfo}
            vbios version:     {self.vbios}
            vram:              {self.vram}
            performance level: {self.perflevel}
            system clock:      {self.sclk}
            memory clock:      {self.mclk}
        ''')


def run(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode == 0:
        return p.stdout.decode('ascii')


def search(pattern, string):
    m = re.search(pattern, string, re.MULTILINE)
    if m is not None:
        return m.group(1)


def get_machine_specs(devicenum):

    cpuinfo    = path('/proc/cpuinfo').read_text()
    meminfo    = path('/proc/meminfo').read_text()
    version    = path('/proc/version').read_text()
    os_release = path('/etc/os-release').read_text()
    rocm_info  = path('/opt/rocm/.info/version-utils').read_text()
    rocm_smi   = run(['rocm-smi', '--showvbios', '--showid', '--showproductname', '--showperflevel', '--showclocks', '--showmeminfo', 'vram'])

    device = rf'^GPU\[{devicenum}\]\s*: '

    hostname    = socket.gethostname()
    cpu         = search(r'^model name\s*: (.*?)$', cpuinfo)
    kernel      = search(r'version (\S*)', version)
    ram         = search(r'MemTotal:\s*(\S*)', meminfo)
    distro      = search(r'PRETTY_NAME="(.*?)"', os_release)
    rocmversion = rocm_info.strip()
    vbios       = search(device + r'VBIOS version: (.*?)$', rocm_smi)
    gpuid       = search(device + r'GPU ID: (.*?)$', rocm_smi)
    deviceinfo  = search(device + r'Card series:\s*(.*?)$', rocm_smi)
    vram        = search(device + r'.... Total Memory .B.: (\d+)$', rocm_smi)
    perflevel   = search(device + r'Performance Level: (.*?)$', rocm_smi)
    mclk        = search(device + r'mclk.*\((.*?)\)$', rocm_smi)
    sclk        = search(device + r'sclk.*\((.*?)\)$', rocm_smi)

    ram = '{:.2f} GiB'.format(float(ram) / 1024**2)
    vram = '{:.2f} GiB'.format(float(vram) / 1024**3)

    bandwidth = None
    if gpuid == '0x66af':
        # radeon7: float: 13.8 TFLOPs, double: 3.46 TFLOPs, 1024 GB/s
        bandwidth = (13.8, 3.46, 1024)

    return MachineSpecs(hostname, cpu, kernel, ram, distro,
                        rocmversion, vbios, gpuid, deviceinfo, vram,
                        perflevel, mclk, sclk, bandwidth)


if __name__ == '__main__':
    print(get_machine_specs(0))
