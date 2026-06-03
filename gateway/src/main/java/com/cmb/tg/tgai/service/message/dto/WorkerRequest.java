package com.cmb.tg.tgai.service.message.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class WorkerRequest {
    @JsonProperty("worker_no")
    private String workerNo;
}
