PROTO_DIR := ../protos
SWIFT_OUT := Sources/EdgeOrchestraProtos

.PHONY: proto-gen build clean

proto-gen:
	@rm -f $(SWIFT_OUT)/*.pb.swift $(SWIFT_OUT)/*.grpc.swift
	protoc \
		-I$(PROTO_DIR) \
		--swift_out=$(SWIFT_OUT) \
		--swift_opt=Visibility=Public \
		--plugin=protoc-gen-grpc-swift=$(shell which protoc-gen-grpc-swift-2) \
		--grpc-swift_out=$(SWIFT_OUT) \
		--grpc-swift_opt=Visibility=Public,Client=true,Server=false \
		$(PROTO_DIR)/edgeorchestra/v1/*.proto
	@if [ -d "$(SWIFT_OUT)/edgeorchestra" ]; then \
		mv $(SWIFT_OUT)/edgeorchestra/v1/*.swift $(SWIFT_OUT)/; \
		rm -rf $(SWIFT_OUT)/edgeorchestra; \
	fi
	@echo "Swift protobuf generation complete."

build:
	swift build

clean:
	swift package clean
