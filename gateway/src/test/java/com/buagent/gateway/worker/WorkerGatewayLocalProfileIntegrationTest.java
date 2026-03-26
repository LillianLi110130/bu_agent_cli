package com.buagent.gateway.worker;

import com.buagent.gateway.GatewayApplication;
import com.buagent.gateway.app.dto.DebugInboundRequest;
import com.buagent.gateway.app.dto.PollRequest;
import com.buagent.gateway.app.dto.PollResponse;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.ActiveProfiles;
import org.springframework.web.context.request.async.DeferredResult;

import static org.junit.jupiter.api.Assertions.*;

@SpringBootTest(classes = GatewayApplication.class)
@ActiveProfiles("local")
class WorkerGatewayLocalProfileIntegrationTest {

    @Autowired
    private WorkerGatewayService workerGatewayService;

    @Test
    void shouldAcceptDebugInboundAndPollMessageWithLocalProfile() {
        workerGatewayService.acceptDebugInbound(new DebugInboundRequest("telegram:999", "hello local"));

        DeferredResult<PollResponse> deferredResult = workerGatewayService.poll(
            new PollRequest("telegram:999", "worker-local-1")
        );

        PollResponse pollResponse = (PollResponse) deferredResult.getResult();
        assertNotNull(pollResponse);
        assertEquals(1, pollResponse.getMessages().size());
        assertEquals("hello local", pollResponse.getMessages().get(0).getContent());
        assertEquals(1L, pollResponse.getMessages().get(0).getEpoch().longValue());
    }
}
