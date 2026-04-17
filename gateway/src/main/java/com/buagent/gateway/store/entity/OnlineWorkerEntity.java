package com.buagent.gateway.store.entity;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class OnlineWorkerEntity {
    private String workerId;
    private String status;
    private LocalDateTime createTime;
    private LocalDateTime updateTime;
}
