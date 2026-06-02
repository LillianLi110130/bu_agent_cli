package com.cmb.tg.tgai.infrastructure.common.config;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.scheduling.annotation.EnableScheduling;
import org.springframework.scheduling.concurrent.ThreadPoolTaskExecutor;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.web.servlet.config.annotation.AsyncSupportConfigurer;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;

import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;

@Configuration
@EnableScheduling
public class ExecutorConfig implements WebMvcConfigurer {

    @Value("${gateway.mvc-async.core-pool-size}")
    private int mvcAsyncCorePoolSize = 16;

    @Value("${gateway.mvc-async.max-pool-size}")
    private int mvcAsyncMaxPoolSize = 128;

    @Value("${gateway.mvc-async.queue-capacity}")
    private int mvcAsyncQueueCapacity = 0;

    @Value("${gateway.mvc-async.thread-name-prefix}")
    private String mvcAsyncThreadNamePrefix = "mvc-async-";

    @Bean
    public ScheduledExecutorService scheduledExecutorService(){
        return Executors.newScheduledThreadPool(10);
    }

    @Bean
    public ThreadPoolTaskExecutor mvcAsyncTaskExecutor() {
        ThreadPoolTaskExecutor taskExecutor = new ThreadPoolTaskExecutor();
        taskExecutor.setCorePoolSize(mvcAsyncCorePoolSize);
        taskExecutor.setMaxPoolSize(mvcAsyncMaxPoolSize);
        taskExecutor.setQueueCapacity(mvcAsyncQueueCapacity);
        taskExecutor.setThreadNamePrefix(mvcAsyncThreadNamePrefix);
        taskExecutor.setAllowCoreThreadTimeOut(true);
        return taskExecutor;
    }

    @Override
    public void configureAsyncSupport(AsyncSupportConfigurer configurer) {
        // StreamingResponseBody is executed by MVC async support; without this Spring falls back
        // to SimpleAsyncTaskExecutor, which creates unbounded threads and is unsafe under load.
        configurer.setTaskExecutor(mvcAsyncTaskExecutor());
        configurer.setDefaultTimeout(600000);
    }
}
