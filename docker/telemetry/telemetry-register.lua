-- 设备密钥幂等注册（方案A：客户端生成对称秘密 + Talos importedApiKeys）
--
-- 请求体（与 Talos import 形状一致）：{raw_key, name, actor_id}
-- 逻辑：限流 → 按 actor_id 查询 → 存在则创建新钥+删旧钥（无缝换绑）/ 不存在则创建
-- 安全注意：知道 actor_id（device_id）的任何人都能换绑该设备身份——遥测场景低风险，已用限流兜底。

local cjson = require("cjson.safe")
local limit_dict = ngx.shared.register_limit

ngx.req.read_body()
local body = ngx.req.get_body_data()
if not body then
    ngx.status = 400
    ngx.say(cjson.encode({error = "empty body"}))
    return
end

local req = cjson.decode(body)
if not req or type(req.raw_key) ~= "string" or #req.raw_key < 32
        or type(req.actor_id) ~= "string" or #req.actor_id == 0 then
    ngx.status = 400
    ngx.say(cjson.encode({error = "raw_key(>=32 chars) and actor_id required"}))
    return
end

-- 每 IP 限流：60 秒 5 次（shared dict 计数器，窗口近似）
local key = "rl:" .. ngx.var.remote_addr
local hits, err = limit_dict:incr(key, 1, 0, 60)
if err then hits = 0 end
if hits > 5 then
    ngx.status = 429
    ngx.say(cjson.encode({error = "rate limited"}))
    return
end

local function talos(method, path, payload)
    local res = ngx.location.capture("/_talos_admin" .. path, {
        method = method,
        body = payload and cjson.encode(payload) or nil,
    })
    local data = res.body and cjson.decode(res.body) or nil
    return res.status, data
end

-- 1. 查询该 actor 是否已绑定（AIP-160 filter）
local filter = ngx.escape_uri(('actor_id="%s"'):format(req.actor_id))
local status, data = talos(ngx.HTTP_GET, "/v2alpha1/admin/importedApiKeys?filter=" .. filter .. "&page_size=1")

-- 2a. 已存在 → 换绑：先创建新钥（旧钥仍有效，无缝）再硬删旧钥
--    （Talos PATCH 只更新 metadata/scopes/rate-limits，不接受 raw_key——返回 200 但凭证不变，实测确认）
if status == 200 and data and data.imported_api_keys and #data.imported_api_keys > 0 then
    local old_key_id = data.imported_api_keys[1].key_id
    local cstatus, cdata = talos(ngx.HTTP_POST, "/v2alpha1/admin/importedApiKeys", {
        raw_key = req.raw_key, name = req.name or "mflowy device", actor_id = req.actor_id,
        ttl = "2592000s",  -- Talos v26 不支持永不过期 key，须给 TTL（30 天；设备可重新注册续期）
    })
    if cstatus == 200 or cstatus == 201 then
        local dstatus = talos(ngx.HTTP_DELETE, "/v2alpha1/admin/importedApiKeys/" .. old_key_id)
        if dstatus ~= 200 and dstatus ~= 201 and dstatus ~= 204 then
            ngx.log(ngx.ERR, string.format("旧钥删除失败（key_id=%s, status=%s），双钥并存至 TTL", old_key_id, dstatus))
        end
        ngx.say(cjson.encode({key_id = cdata and cdata.key_id or nil, bound = "updated"}))
        return
    end
    ngx.status = 502
    ngx.say(cjson.encode({error = "talos import failed", status = cstatus}))
    return
end

-- 2b. 创建
local cstatus, cdata = talos(ngx.HTTP_POST, "/v2alpha1/admin/importedApiKeys", {
    raw_key = req.raw_key, name = req.name or "mflowy device", actor_id = req.actor_id,
    ttl = "2592000s",  -- Talos v26 不支持永不过期 key，须给 TTL（30 天；设备可重新注册续期）
})
if cstatus == 200 or cstatus == 201 then
    ngx.say(cjson.encode({key_id = cdata and cdata.key_id or nil, bound = "created"}))
else
    ngx.status = 502
    ngx.say(cjson.encode({error = "talos import failed", status = cstatus}))
end
