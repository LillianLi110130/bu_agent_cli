package com.cmb.tg.tgai.controller.message;

import com.cmb.tg.tgai.service.message.dto.SubmitWebMessageRequest;
import com.cmb.tg.tgai.service.message.dto.SubmitWebMessageResponse;
import com.cmb.tg.tgai.service.message.dto.WebWorkerSummaryResponse;
import com.cmb.tg.tgai.service.worker.WebConsoleService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

@RestController
@RequestMapping("/web-console")
@RequiredArgsConstructor
public class WebConsoleController {

    private final WebConsoleService webConsoleService;

    @GetMapping("/workers/{workerId}")
    public WebWorkerSummaryResponse workerSummary(@PathVariable("workerId") String workerId) {
        return webConsoleService.getWorkerSummary();
    }

    @PostMapping("/messages")
    public SubmitWebMessageResponse submitMessage(@RequestBody SubmitWebMessageRequest request) {
        return webConsoleService.submitMessage(request);
    }

    @GetMapping(path = "/workers/{workerId}/events", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public SseEmitter workerEvents(@PathVariable("workerId") String workerId) {
        return webConsoleService.stream();
    }
}
