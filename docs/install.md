# Installation

Every release provides ready-to-use packages for Linux x64 and Windows x64 on
its [GitHub Release](https://github.com/MotohiroOGAWA/msentity/releases) page.
The examples use version 1.2.2, the latest release when this page was written;
replace `1.2.2` with the version you need. The newest version is marked *Latest*
on the [releases page](https://github.com/MotohiroOGAWA/msentity/releases/latest),
or run `gh release view --repo MotohiroOGAWA/msentity`.

| Platform | Python wheel tag | Java / C# / C platform |
|---|---|---|
| Linux x64 (glibc ≥ 2.35) | `manylinux_2_35_x86_64` | `linux-x64` |
| Windows x64 | `win_amd64` | `win-x64` |

Release 1.2.1 is an exception: its Linux packages need glibc ≥ 2.41 (wheel tag `manylinux_2_41_x86_64`). <!-- fixed -->
On older Linux, use 1.2.0 or a later release. <!-- fixed -->

## C

Download `msentity-c-<version>-<platform>.zip` from the release page, for
example [`v1.2.2`](https://github.com/MotohiroOGAWA/msentity/releases/tag/v1.2.2):

```text
msentity-c-1.2.2-linux-x64/
├── include/msentity/c/msentity_c.h
├── lib/libmsentity_c.so
└── LICENSE
```

```console
cc app.c -I msentity-c-1.2.2-linux-x64/include \
  -L msentity-c-1.2.2-linux-x64/lib -lmsentity_c
```

Windows provides `msentity_c.dll` and `msentity_c.lib`.

## C++

The C++ API exposes Arrow types, so C++ projects build msentity from source
against their own Arrow, Parquet, and HDF5. Pin a release tag:

```cmake
include(FetchContent)
FetchContent_Declare(msentity
  GIT_REPOSITORY https://github.com/MotohiroOGAWA/msentity.git
  GIT_TAG v1.2.2)
FetchContent_MakeAvailable(msentity)

target_link_libraries(app PRIVATE msentity::msentity)
```

Required build dependencies are CMake ≥ 3.21, a C++20 compiler, Arrow with
Parquet, HDF5 with its high-level API, and nlohmann/json. With vcpkg, this
repository's `vcpkg.json` installs them automatically.

To install a selected release system-wide:

```console
git clone --branch v1.2.2 https://github.com/MotohiroOGAWA/msentity.git
cmake -S msentity -B msentity/build -DCMAKE_BUILD_TYPE=Release
cmake --build msentity/build
cmake --install msentity/build --prefix /usr/local
```

```cmake
find_package(msentity REQUIRED)
target_link_libraries(app PRIVATE msentity::msentity)
```

## Python (≥ 3.10)

The wheels are attached to each GitHub Release. Point pip at the release page
with `--find-links`; it picks the wheel for the current platform:

```console
pip install msentity==1.2.2 \
  --find-links https://github.com/MotohiroOGAWA/msentity/releases/expanded_assets/v1.2.2
pip install "msentity[arrow]==1.2.2" \
  --find-links https://github.com/MotohiroOGAWA/msentity/releases/expanded_assets/v1.2.2
```

Use the same version in `==` and in the URL. (Earlier instructions used a
GitHub Pages index, `https://motohiroogawa.github.io/msentity/`; it is no longer
updated and lists releases up to 1.2.0 only.) <!-- fixed -->

```python
import msentity as ms
ds = ms.load("example.msp")
```

## Java (≥ 22)

JitPack serves the JAR attached to a release tag:

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
    <version>v1.2.2</version>
  </dependency>
</dependencies>
```

Gradle: `implementation 'com.github.MotohiroOGAWA:msentity:v1.2.2'` with
`maven { url 'https://jitpack.io' }`. The JAR (`msentity-1.2.2.jar`) is also
attached to the release. Run with `--enable-native-access=ALL-UNNAMED`.

## C# (.NET ≥ 8)

Download `MsEntity.<version>.nupkg` from the release into a local package
folder and use it as a package source:

```console
mkdir -p ~/nupkg
gh release download v1.2.2 --repo MotohiroOGAWA/msentity --pattern 'MsEntity.*.nupkg' --dir ~/nupkg
dotnet add package MsEntity --version 1.2.2 --source ~/nupkg
```

The package contains native libraries under `runtimes/<platform>/native/`.

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

Releases are built and tested on Ubuntu 22.04 (Linux x64, glibc 2.35) and on
Windows (x64), and every package is attached to the GitHub Release of the tag
`v<version>`; the version in `CMakeLists.txt` equals the tag. The maintainers'
scripts are in
[msentity-spectrum-viewer/scripts/release-all](https://github.com/MotohiroOGAWA/msentity-spectrum-viewer/tree/main/scripts/release-all).
