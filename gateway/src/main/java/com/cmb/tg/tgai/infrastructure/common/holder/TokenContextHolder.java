package com.cmb.tg.tgai.infrastructure.common.holder;

public final class TokenContextHolder {

    private static final String CURRENT_USER_ID = "userNo";
    private static final String CURRENT_OPEN_ID = "openId";
    private static final String CURRENT_YST_ID = "ystId";

    private TokenContextHolder() {
    }

    public static String getUserIdOfCurrentUser() {
        return CURRENT_USER_ID;
    }

    public static String getOpenIdOfCurrentUser() {
        return CURRENT_OPEN_ID;
    }

    public static String getYstIdOfCurrentUser() {
        return CURRENT_YST_ID;
    }
}
