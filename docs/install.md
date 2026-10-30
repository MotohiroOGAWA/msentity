# Installation

Every release provides ready-to-use packages for Linux x64, Windows x64, and
macOS arm64. When a version is omitted, the commands below install or download
the latest release. Pin a version for reproducible applications.

| Platform | Python wheel tag | Java / C# / C platform |
|---|---|---|
| Linux x64 (glibc ≥ 2.35) | `manylinux_2_35_x86_64` | `linux-x64` |
| Windows x64 | `win_amd64` | `win-x64` |
| macOS arm64 (macOS ≥ 14) | `macosx_14_0_arm64` | `osx-arm64` |

## C

Download `msentity-c-<version>-<platform>.zip` from the
[latest release](https://github.com/MotohiroOGAWA/msentity/releases/latest).
For a fixed version, use its release page, such as
[`v0.1.0`](https://github.com/MotohiroOGAWA/msentity/releases/tag/v0.1.0).

```text
msentity-c-0.1.0-linux-x64/
├── include/msentity/c/msentity_c.h
├── lib/libmsentity_c.so
└── LICENSE
```

```console
cc app.c -I msentity-c-0.1.0-linux-x64/include \
  -L msentity-c-0.1.0-linux-x64/lib -lmsentity_c
```

Windows provides `msentity_c.dll` and `msentity_c.lib`; macOS provides
`libmsentity_c.dylib`.

## C++

The C++ API exposes Arrow types, so C++ projects build msentity from source
against their own Arrow, Parquet, and HDF5. Pin a release tag in production:

```cmake
include(FetchContent)
FetchContent_Declare(msentity
  GIT_REPOSITORY https://github.com/MotohiroOGAWA/msentity.git
  GIT_TAG v0.1.0)
FetchContent_MakeAvailable(msentity)

target_link_libraries(app PRIVATE msentity::msentity)
```

Required build dependencies are CMake ≥ 3.21, a C++20 compiler, Arrow with
Parquet, HDF5 with its high-level API, and nlohmann/json. With vcpkg, this
repository's `vcpkg.json` installs them automatically.

To install a selected release system-wide:

```console
git clone --branch v0.1.0 https://github.com/MotohiroOGAWA/msentity.git
cmake -S msentity -B msentity/build -DCMAKE_BUILD_TYPE=Release
cmake --build msentity/build
cmake --install msentity/build --prefix /usr/local
```

```cmake
find_package(msentity REQUIRED)
target_link_libraries(app PRIVATE msentity::msentity)
```

## Python (≥ 3.10)

PyPI selects the wheel for the current platform. With no version, pip installs
the latest release:

```console
pip install msentity
pip install "msentity[arrow]"       # also installs pyarrow and pandas
```

To install a specific version:

```console
pip install msentity==0.1.0
```

```python
import msentity as ms
ds = ms.load("example.msp")
```

## Java (≥ 22)

JitPack resolves either the latest release or an explicit tag.

```xml
<repositories>
  <repository>
    <id>jitpack.io</id>
    <url>https://jitpack.io</url>
  </repository>
</repositories>
<dependencies>
  <dependency>
    <groupId>com.github.MotohiroOGAWA</groupId>
    <artifactId>msentity</artifactId>
    <version>[0,)</version>
  </dependency>
</dependencies>
```

The Maven range selects the newest release already built by JitPack. Use
`<version>v0.1.0</version>` to pin it. Gradle users can use
`com.github.MotohiroOGAWA:msentity:latest.release`, or replace the last
component with `v0.1.0`. Run with `--enable-native-access=ALL-UNNAMED`.

## C# (.NET ≥ 8)

Download `MsEntity.<version>.nupkg` from the
[latest release](https://github.com/MotohiroOGAWA/msentity/releases/latest)
into a local package folder. Omitting the version selects the newest package
available in that source:

```console
dotnet add package MsEntity --source ~/nupkg
dotnet add package MsEntity --version 0.1.0 --source ~/nupkg
```

The second form pins a version. The package contains native libraries under
`runtimes/<platform>/native/`.

## From a source checkout (development)

```console
git clone https://github.com/MotohiroOGAWA/msentity.git && cd msentity
cmake -S . -B build -G Ninja && cmake --build build
ctest --test-dir build

pip install -e bindings/python
(cd bindings/java && mvn install)
dotnet add reference bindings/csharp/MsEntity/MsEntity.csproj
```

## How releases are made

Pushing a tag `v<version>` equal to the version in `CMakeLists.txt` runs the
[release workflow](../.github/workflows/release.yml). It builds and tests all
platform packages, publishes wheels to PyPI, and attaches every artifact to a
GitHub Release. PyPI uses a trusted publisher configured for this repository's
`pypi` environment.
