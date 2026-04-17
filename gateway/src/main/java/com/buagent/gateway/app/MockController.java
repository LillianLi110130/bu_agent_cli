package com.buagent.gateway.app;

import com.buagent.gateway.app.dto.MessageRequest;
import com.buagent.gateway.app.dto.SimpleOkResponse;
import com.buagent.gateway.worker.WorkerGatewayService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/mock")
public class MockController {

    private static final Logger logger = LoggerFactory.getLogger(MockController.class);

    private final WorkerGatewayService workerGatewayService;

    public MockController(WorkerGatewayService workerGatewayService) {
        this.workerGatewayService = workerGatewayService;
    }

    @PostMapping("/messages")
    public SimpleOkResponse messages(@RequestBody MessageRequest request) {
        logger.info(
            "Received mock message request. workerId={}, contentLength={}",
            request.getWorkerId(),
            request.getContent() == null ? 0 : request.getContent().length()
        );
        SimpleOkResponse response = workerGatewayService.acceptMockMessage(request);
        logger.info("Completed mock message request. workerId={}, ok={}", request.getWorkerId(), response.getOk());
        return response;
    }
}
