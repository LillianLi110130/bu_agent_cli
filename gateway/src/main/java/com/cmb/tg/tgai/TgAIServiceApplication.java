package com.cmb.tg.tgai;

import org.mybatis.spring.annotation.MapperScan;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.scheduling.annotation.EnableScheduling;

@SpringBootApplication
@MapperScan("com.cmb.tg.tgai")
@EnableScheduling
public class TgAIServiceApplication {

    public static void main(String[] args) {
        SpringApplication.run(TgAIServiceApplication.class, args);
    }
}
