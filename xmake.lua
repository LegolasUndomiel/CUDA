-- 模式添加
add_rules("mode.debug", "mode.release")

-- 设置编码格式
set_encodings("utf-8")

-- 设置编译标准
set_languages("c++11")

-- 自动更新Visual Studio解决方案
add_rules("plugin.vsxmake.autoupdate")

-- 设置自定义清理脚本
on_clean(function (target)
    print("All Files Deleted")
    -- 删除所有文件
    os.rm("$(buildir)")
    os.rm(target:targetdir())
end)

set_targetdir("bin")

target("omptest")
    set_kind("binary")

    add_files("src/omptest.cc")

    if is_plat("linux") then
        add_cxxflags("-fopenmp")

        add_ldflags("-fopenmp")
    elseif is_plat("windows") then
        add_cxxflags("/openmp")

        add_ldflags("/openmp")
    end

target("CUDAAdd")
    set_kind("binary")

    add_files("src/CUDAAdd.cu")

    add_cugencodes("native")

target("MandelbrotSetC")
    set_kind("binary")

    add_files("src/MandelbrotSet.c")

target("MandelbrotSetCPP")
    set_kind("binary")
    add_ldflags("-pthread")

    add_files("src/MandelbrotSet.cpp")

target("MandelbrotSetCUDA")
    set_kind("binary")

    add_files("src/MandelbrotSet.cu")

    add_cugencodes("native")

target("CalIntegral")
    set_kind("binary")

    add_includedirs("$(env CUDA_PATH)/include")
    add_files("src/CalIntegral.cu")

    add_cugencodes("sm_86")

    if is_plat("linux") then
        add_cxxflags("-fopenmp")

        add_ldflags("-fopenmp")

        add_cuflags("-Xcompiler -fopenmp")

        add_culdflags("-Xcompiler -fopenmp")
    elseif is_plat("windows") then
        add_cxxflags("/openmp")

        add_ldflags("/openmp")

        add_cuflags("-Xcompiler /openmp")

        add_culdflags("-Xcompiler /openmp")
    end
