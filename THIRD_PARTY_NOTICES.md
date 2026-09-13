# Third-Party Notices

Caustica's project-owned code is licensed under `LGPL-3.0-or-later`. This file
documents third-party components and license boundaries that are not changed by
Caustica's license.

## NVIDIA DLSS / NGX SDK

Caustica can build and distribute release artifacts that include NVIDIA DLSS/NGX
SDK runtime components, including DLSS Ray Reconstruction and Frame Generation
libraries. These NVIDIA components are proprietary third-party software and are
not licensed under the LGPL.

The NVIDIA SDK components remain subject to the NVIDIA RTX SDKs license:

<https://github.com/NVIDIA/DLSS/blob/main/LICENSE.txt>

The LGPL license grant for Caustica does not grant rights to NVIDIA SDK
components. Redistribution and use of those components must comply with
NVIDIA's license terms.

This software contains source code provided by NVIDIA Corporation.

Bundled NVIDIA SDK runtime libraries may include files matching:

- `caustica/natives/windows-x64/nvngx_dlssd.dll`
- `caustica/natives/windows-x64/nvngx_dlssg.dll`
- `caustica/natives/linux-x64/libnvidia-ngx-dlssd.so*`
- `caustica/natives/linux-x64/libnvidia-ngx-dlssg.so*`

Caustica's `ngxshim` native library is project-owned glue code and follows
Caustica's project license unless otherwise noted.

## Qualcomm Snapdragon Game Super Resolution 2 (SGSR2)

The reconstruction chain's SGSR2 upscaler (`shaders/pipelines/sgsr/`) is a port of Qualcomm's
Snapdragon Game Super Resolution 2 reference compute shaders to this renderer's Slang pipeline style,
with this project's depth, motion-vector and guide conventions adapted as described in those files.

```
Copyright (c) 2024, Qualcomm Innovation Center, Inc. All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
```

Redistribution and use in source and binary forms, with or without modification, are permitted provided
that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the
   following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and
   the following disclaimer in the documentation and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or
   promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

---

## NVIDIA Real-time Denoisers (NRD)

Caustica can build and distribute release artifacts that include NVIDIA Real-time
Denoisers, linked statically into the `caustica_nrd` shim together with its
SPIR-V shader blobs. NRD is proprietary third-party software and is not licensed
under the LGPL.

NRD remains subject to the NVIDIA RTX SDKs license:

<https://github.com/NVIDIA-RTX/NRD/blob/main/LICENSE.txt>

The LGPL license grant for Caustica does not grant rights to NRD. Redistribution
and use of NRD must comply with NVIDIA's license terms.

This software contains source code provided by NVIDIA Corporation.

The NRD sources are never part of this repository. The shim's build fetches them
from NVIDIA's own distribution channel at configure time
(`https://github.com/NVIDIA-RTX/NRD.git`, pinned to a release tag), and only the
resulting object code is shipped. Bundled NRD runtime libraries may include files
matching:

- `caustica/natives/windows-x64/caustica_nrd.dll`
- `caustica/natives/linux-x64/libcaustica_nrd.so`

Caustica's `caustica_nrd` shim is project-owned glue code and follows Caustica's
project license unless otherwise noted. Its host-side shader ports
(`shaders/pipelines/nrd_frontend`, `shaders/pipelines/nrd_backend`) reimplement
encoding helpers described in NRD's `NRD.hlsli` and carry the same notice.
