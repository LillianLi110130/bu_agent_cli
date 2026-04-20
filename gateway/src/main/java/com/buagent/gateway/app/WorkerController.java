package com.buagent.gateway.app;

import com.buagent.gateway.app.dto.CompleteRequest;
import com.buagent.gateway.app.dto.SimpleOkResponse;
import com.buagent.gateway.app.dto.WorkerRequest;
import com.buagent.gateway.worker.WorkerGatewayService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

@RestController
@RequestMapping("/api/worker")
public class WorkerController {

    private static final Logger logger = LoggerFactory.getLogger(WorkerController.class);

    private final WorkerGatewayService workerGatewayService;

    public WorkerController(WorkerGatewayService workerGatewayService) {
        this.workerGatewayService = workerGatewayService;
    }

    @PostMapping("/online")
    public SimpleOkResponse online(@RequestBody WorkerRequest request) {
        SimpleOkResponse response = workerGatewayService.online(request);
        return response;
    }

    @PostMapping("/offline")
    public SimpleOkResponse offline(@RequestBody WorkerRequest request) {
        SimpleOkResponse response = workerGatewayService.offline(request);
        return response;
    }

    @GetMapping(path = "/stream", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public SseEmitter stream(@RequestParam("worker_id") String workerId) {
        return workerGatewayService.stream(workerId);
    }

    @PostMapping("/complete")
    public SimpleOkResponse complete(@RequestBody CompleteRequest request) {
        logger.info(
            "Received complete request. workerId={}, finalContentLength={}",
            request.getWorkerId(),
            request.getFinalContent() == null ? 0 : request.getFinalContent().length()
        );
        SimpleOkResponse response = workerGatewayService.complete(request);
        logger.info("Completed complete request. workerId={}, ok={}", request.getWorkerId(), response.getOk());
        return response;
    }
}
