package com.buagent.gateway.app.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class UploadAttachmentRequest {
    @JsonProperty("session_key")
    private String sessionKey;

    @JsonProperty("worker_id")
    private String workerId;

    @JsonProperty("file_name")
    private String fileName;

    @JsonProperty("mime_type")
    private String mimeType;

    @JsonProperty("file_size")
    private Long fileSize;

    @JsonProperty("file_content_base64")
    private String fileContentBase64;
}
