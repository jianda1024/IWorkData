package com.dake.local;

import java.io.IOException;
import java.io.UnsupportedEncodingException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Base64;

public class Local {

    public static void main(String[] args) throws Exception {
        Path jar = Path.of("E:\\Downloads\\baidu\\commons-lang3-3.22.0.jar");
        Path txt = Path.of("E:\\Downloads\\baidu\\commons-lang3-3.22.0.txt");
        Path outJar = Path.of("E:\\Downloads\\baidu\\newcommons-lang3-3.22.0.jar");


        // 编码：jar -> txt
        byte[] bytes = Files.readAllBytes(jar);
        String s = Base64.getEncoder().encodeToString(bytes);
        Files.writeString(txt, s);

        // 解码：txt -> jar
        String read = Files.readString(txt);
        byte[] decoded = Base64.getDecoder().decode(read);
        Files.write(outJar, decoded);

        System.out.println("编码完成，文件大小: " + Files.size(txt) + " 字节");
    }
}
