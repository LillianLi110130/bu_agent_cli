package com.buagent.gateway.app.dto;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;

import java.util.Collections;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class DtoJsonBindingTest {

    private final ObjectMapper objectMapper = new ObjectMapper();

    @Test
    void shouldDeserializeWorkerRequestsFromSnakeCaseJson() throws Exception {
        PollRequest pollRequest = objectMapper.readValue(
            "{\"session_key\":\"telegram:123\",\"worker_id\":\"worker-1\"}",
            PollRequest.class
        );
        RenewRequest renewRequest = objectMapper.readValue(
            "{\"session_key\":\"telegram:123\",\"worker_id\":\"worker-1\",\"delivery_id\":\"d-1\"}",
            RenewRequest.class
        );
        CompleteRequest completeRequest = objectMapper.readValue(
            "{\"session_key\":\"telegram:123\",\"worker_id\":\"worker-1\",\"delivery_id\":\"d-1\",\"final_content\":\"done\"}",
            CompleteRequest.class
        );
        SendTextRequest sendTextRequest = objectMapper.readValue(
            "{\"session_key\":\"telegram:123\",\"worker_id\":\"worker-1\",\"text\":\"hello proactive\"}",
            SendTextRequest.class
        );
        UploadAttachmentRequest uploadAttachmentRequest = objectMapper.readValue(
            "{\"session_key\":\"telegram:123\",\"worker_id\":\"worker-1\",\"file_name\":\"report.pdf\",\"mime_type\":\"application/pdf\",\"file_size\":123,\"file_content_base64\":\"cGRm\"}",
            UploadAttachmentRequest.class
        );
        DebugInboundRequest debugInboundRequest = objectMapper.readValue(
            "{\"session_key\":\"telegram:123\",\"content\":\"hello\"}",
            DebugInboundRequest.class
        );

        assertEquals("telegram:123", pollRequest.getSessionKey());
        assertEquals("worker-1", pollRequest.getWorkerId());
        assertEquals("telegram:123", renewRequest.getSessionKey());
        assertEquals("worker-1", renewRequest.getWorkerId());
        assertEquals("d-1", renewRequest.getDeliveryId());
        assertEquals("telegram:123", completeRequest.getSessionKey());
        assertEquals("worker-1", completeRequest.getWorkerId());
        assertEquals("d-1", completeRequest.getDeliveryId());
        assertEquals("done", completeRequest.getFinalContent());
        assertEquals("telegram:123", sendTextRequest.getSessionKey());
        assertEquals("worker-1", sendTextRequest.getWorkerId());
        assertEquals("hello proactive", sendTextRequest.getText());
        assertEquals("telegram:123", uploadAttachmentRequest.getSessionKey());
        assertEquals("worker-1", uploadAttachmentRequest.getWorkerId());
        assertEquals("report.pdf", uploadAttachmentRequest.getFileName());
        assertEquals("application/pdf", uploadAttachmentRequest.getMimeType());
        assertEquals(123L, uploadAttachmentRequest.getFileSize().longValue());
        assertEquals("cGRm", uploadAttachmentRequest.getFileContentBase64());
        assertEquals("telegram:123", debugInboundRequest.getSessionKey());
        assertEquals("hello", debugInboundRequest.getContent());
    }

    @Test
    void shouldSerializePollResponseUsingSnakeCaseJson() throws Exception {
        PollResponse pollResponse = new PollResponse(
            Collections.singletonList(
                new PollMessageDto(101L, "d-1", 3L, "hello")
            )
        );

        JsonNode root = objectMapper.readTree(objectMapper.writeValueAsString(pollResponse));
        JsonNode message = root.get("messages").get(0);

        assertTrue(message.has("message_id"));
        assertTrue(message.has("delivery_id"));
        assertEquals(101L, message.get("message_id").asLong());
        assertEquals("d-1", message.get("delivery_id").asText());
        assertEquals(3L, message.get("epoch").asLong());
        assertEquals("hello", message.get("content").asText());
    }

    @Test
    void shouldDeserializeWorkerMinimalProtocolJson() throws Exception {
        PollRequest pollRequest = objectMapper.readValue(
            "{\"worker_id\":\"worker-1\"}",
            PollRequest.class
        );
        CompleteRequest completeRequest = objectMapper.readValue(
            "{\"worker_id\":\"worker-1\",\"final_content\":\"done\"}",
            CompleteRequest.class
        );
        SendTextRequest sendTextRequest = objectMapper.readValue(
            "{\"worker_id\":\"worker-1\",\"text\":\"hello proactive\"}",
            SendTextRequest.class
        );
        UploadAttachmentRequest uploadAttachmentRequest = objectMapper.readValue(
            "{\"worker_id\":\"worker-1\",\"file_name\":\"report.pdf\",\"mime_type\":\"application/pdf\",\"file_size\":123,\"file_content_base64\":\"cGRm\"}",
            UploadAttachmentRequest.class
        );

        assertNull(pollRequest.getSessionKey());
        assertEquals("worker-1", pollRequest.getWorkerId());
        assertNull(completeRequest.getSessionKey());
        assertEquals("worker-1", completeRequest.getWorkerId());
        assertNull(completeRequest.getDeliveryId());
        assertEquals("done", completeRequest.getFinalContent());
        assertNull(sendTextRequest.getSessionKey());
        assertEquals("worker-1", sendTextRequest.getWorkerId());
        assertEquals("hello proactive", sendTextRequest.getText());
        assertNull(uploadAttachmentRequest.getSessionKey());
        assertEquals("worker-1", uploadAttachmentRequest.getWorkerId());
        assertEquals("report.pdf", uploadAttachmentRequest.getFileName());
        assertEquals("application/pdf", uploadAttachmentRequest.getMimeType());
        assertEquals(123L, uploadAttachmentRequest.getFileSize().longValue());
        assertEquals("cGRm", uploadAttachmentRequest.getFileContentBase64());
    }
}
