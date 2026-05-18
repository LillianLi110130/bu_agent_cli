package com.buagent.gateway.app.dto;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class WebWorkerSummaryResponse {
    private String workerId;
    private Boolean isOnline;
}
