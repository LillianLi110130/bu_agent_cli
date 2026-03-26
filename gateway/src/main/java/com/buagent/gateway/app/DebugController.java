package com.buagent.gateway.app;

import com.buagent.gateway.app.dto.DebugInboundRequest;
import com.buagent.gateway.app.dto.SimpleOkResponse;
import com.buagent.gateway.worker.WorkerGatewayService;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/debug")
public class DebugController {

    private final WorkerGatewayService workerGatewayService;

    public DebugController(WorkerGatewayService workerGatewayService) {
        this.workerGatewayService = workerGatewayService;
    }

    @PostMapping("/inbound")
    public SimpleOkResponse inbound(@RequestBody DebugInboundRequest request) {
        return workerGatewayService.acceptDebugInbound(request);
    }
}
