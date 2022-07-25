# Definitions
build_output = runtime/deepurfold
runtime_fullpath = $(realpath runtime)
build_tool = runtime-container.DONE
#git_commit ?= $(shell git log --pretty=oneline -n 1 -- ../DeepUrfold | cut -f1 -d " ")
name = edraizen/deepurfold
tag = 0.0.1

# Steps
build:
	docker build -t ${name}:${tag} -f ./Dockerfile ..
	-docker rmi -f ${name}:latest
	docker tag ${name}:${tag} ${name}:latest
	touch ${build_tool}

push: build
	# Requires ~/.dockercfg
	docker push ${name}:${tag}
	docker push ${name}:latest

# test: build
# 	python test.py

clean:
	-rm ${build_tool}
	-rm ${build_output}
