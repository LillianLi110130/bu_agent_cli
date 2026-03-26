package com.buagent.gateway.session;

import com.buagent.gateway.app.dto.PollResponse;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.springframework.web.context.request.async.DeferredResult;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class PollWaiter {
    private String workerId;
    private DeferredResult<PollResponse> deferredResult;
}
