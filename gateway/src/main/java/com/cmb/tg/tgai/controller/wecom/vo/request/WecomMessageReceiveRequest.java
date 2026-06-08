package com.cmb.tg.tgai.controller.wecom.vo.request;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Getter;
import lombok.Setter;
import lombok.ToString;

import javax.validation.constraints.NotBlank;
import javax.validation.constraints.NotNull;

@Getter
@Setter
@ToString
public class WecomMessageReceiveRequest {

    @JsonProperty("msgId")
    @NotBlank
    private String msgId;

    @JsonProperty("fromUser")
    @NotBlank
    private String fromUser;

    @JsonProperty("toList")
    private String toList;

    @JsonProperty("roomId")
    private String roomId;

    @JsonProperty("msgType")
    private String msgType;

    @JsonProperty("msgTimeLong")
    @NotNull
    private Long msgTimeLong;

    @JsonProperty("msgBody")
    private String msgBody;

    @JsonProperty("acsKey")
    private String acsKey;

    @JsonProperty("wthrFromCm")
    private String wthrFromCm;
}
