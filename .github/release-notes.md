Install msentity @VERSION@ for your language. The packages contain a self-contained native library for **Linux x64** (glibc ≥ 2.35) and **Windows x64** — no C++ toolchain is needed.

### C
`msentity-c-@VERSION@-<platform>.zip`: `include/msentity/c/msentity_c.h` and the shared library.

### C++
Build from source with CMake (Arrow, Parquet, HDF5 required); see [docs/install.md](https://github.com/MotohiroOGAWA/msentity/blob/@TAG@/docs/install.md).

### Python (≥ 3.10)
```console
pip install msentity --find-links https://motohiroogawa.github.io/msentity/

# Or install this exact release:
pip install msentity==@VERSION@ --find-links https://motohiroogawa.github.io/msentity/
```

### Java (≥ 22)
Maven — add the JitPack repository and the dependency:
```xml
<repositories><repository><id>jitpack.io</id><url>https://jitpack.io</url></repository></repositories>
<dependency><groupId>com.github.MotohiroOGAWA</groupId><artifactId>msentity</artifactId><version>@TAG@</version></dependency>
```
Gradle: `maven { url 'https://jitpack.io' }` and `implementation 'com.github.MotohiroOGAWA:msentity:@TAG@'`.
Run with `--enable-native-access=ALL-UNNAMED`. The JAR is also attached below (`msentity-@VERSION@.jar`).

### C# (.NET ≥ 8)
Download `MsEntity.@VERSION@.nupkg` below into a folder, then:
```console
dotnet add package MsEntity --version @VERSION@ --source <folder>
```
