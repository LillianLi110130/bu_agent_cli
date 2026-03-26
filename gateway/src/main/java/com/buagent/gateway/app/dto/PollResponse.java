package com.buagent.gateway.app.dto;

import java.util.Collections;
import java.util.List;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class PollResponse {
    private List<PollMessageDto> messages;

    public static PollResponse empty() {
        return new PollResponse(Collections.emptyList());
    }
}
