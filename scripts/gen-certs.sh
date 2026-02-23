#!/usr/bin/env bash
set -euo pipefail

CERT_DIR="certs"
DAYS_CA=3650
DAYS_CERT=825
KEY_SIZE=4096

if [ -d "$CERT_DIR" ]; then
    read -rp "Directory '$CERT_DIR' already exists. Overwrite? [y/N] " answer
    if [[ ! "$answer" =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi
    rm -rf "$CERT_DIR"
fi

mkdir -p "$CERT_DIR"

echo "==> Generating CA key and certificate..."
openssl genrsa -out "$CERT_DIR/ca.key" "$KEY_SIZE" 2>/dev/null
openssl req -new -x509 -key "$CERT_DIR/ca.key" -out "$CERT_DIR/ca.crt" \
    -days "$DAYS_CA" -subj "/CN=EdgeOrchestra CA"

echo "==> Generating server key and certificate..."
openssl genrsa -out "$CERT_DIR/server.key" "$KEY_SIZE" 2>/dev/null
openssl req -new -key "$CERT_DIR/server.key" -out "$CERT_DIR/server.csr" \
    -subj "/CN=EdgeOrchestra Server"
cat > "$CERT_DIR/server-ext.cnf" <<EOF
subjectAltName = DNS:localhost, DNS:orchestrator, IP:127.0.0.1
EOF
openssl x509 -req -in "$CERT_DIR/server.csr" -CA "$CERT_DIR/ca.crt" -CAkey "$CERT_DIR/ca.key" \
    -CAcreateserial -out "$CERT_DIR/server.crt" -days "$DAYS_CERT" \
    -extfile "$CERT_DIR/server-ext.cnf" 2>/dev/null

echo "==> Generating client key and certificate..."
openssl genrsa -out "$CERT_DIR/client.key" "$KEY_SIZE" 2>/dev/null
openssl req -new -key "$CERT_DIR/client.key" -out "$CERT_DIR/client.csr" \
    -subj "/CN=EdgeOrchestra Client"
openssl x509 -req -in "$CERT_DIR/client.csr" -CA "$CERT_DIR/ca.crt" -CAkey "$CERT_DIR/ca.key" \
    -CAcreateserial -out "$CERT_DIR/client.crt" -days "$DAYS_CERT" 2>/dev/null

# Clean up temporary files
rm -f "$CERT_DIR"/*.csr "$CERT_DIR"/*.cnf "$CERT_DIR"/*.srl

echo "==> Certificates generated in '$CERT_DIR/':"
ls -la "$CERT_DIR"/
echo ""
echo "Files:"
echo "  $CERT_DIR/ca.key, $CERT_DIR/ca.crt         — CA (self-signed)"
echo "  $CERT_DIR/server.key, $CERT_DIR/server.crt  — Server (SAN: localhost, orchestrator, 127.0.0.1)"
echo "  $CERT_DIR/client.key, $CERT_DIR/client.crt  — Client (for control-plane & iOS)"
