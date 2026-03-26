package com.buagent.gateway.store.entity;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class SessionStateEntity {
    private String sessionKey;
    private Long currentEpoch;
}
