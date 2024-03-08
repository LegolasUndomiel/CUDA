-- 工程名称
set_project("CUDA")

-- 模式添加
add_rules("mode.debug", "mode.release")

-- 设置编译标准
set_languages("c++11")

-- 自动更新Visual Studio解决方案
add_rules("plugin.vsxmake.autoupdate")

-- 设置自定义清理脚本
on_clean(function (target)
    print("All Files Deleted")
    -- 删除所有文件
    os.rm("$(buildir)")
end)

target("CUDAAdd")
    set_kind("binary")

    add_files("src/CUDAAdd.cu")
