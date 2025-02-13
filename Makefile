PYTHON?=python
VERSION?=$(shell sed -ne "s|^__version__\s*=\s*'\([^']*\)'.*|\1|p" src/quicktions.pyx)
PACKAGE=quicktions
WITH_CYTHON := $(shell python -c 'from Cython.Build import cythonize' 2>/dev/null && echo "--with-cython")
PYTHON_WHEEL_BUILD_VERSION := "cp*"

MANYLINUX_IMAGES= \
    manylinux1_x86_64 \
    manylinux1_i686 \
    manylinux_2_24_x86_64 \
    manylinux_2_24_i686 \
    manylinux_2_24_aarch64 \
    manylinux_2_28_x86_64 \
    manylinux_2_34_x86_64 \
    manylinux_2_28_aarch64 \
    manylinux_2_34_aarch64 \
    musllinux_1_1_x86_64 \
    musllinux_1_2_x86_64

.PHONY: all local sdist test clean realclean wheel_manylinux

all:  local

local:
	${PYTHON} setup.py build_ext --inplace $(WITH_CYTHON)

sdist: dist/$(PACKAGE)-$(VERSION).tar.gz

dist/$(PACKAGE)-$(VERSION).tar.gz:
	$(PYTHON) setup.py sdist $(WITH_CYTHON)

testslow: local
	PYTHONPATH=src $(PYTHON) src/test_fractions.py

test: local
	PYTHONPATH=src $(PYTHON) src/test_fractions.py --fast

clean:
	rm -fr build src/*.so

realclean: clean
	rm -fr src/*.c src/*.html

qemu-user-static:
	docker run --rm --privileged hypriot/qemu-register

wheel:
	$(PYTHON) setup.py bdist_wheel

wheel_manylinux: sdist $(addprefix wheel_,$(MANYLINUX_IMAGES))
$(addprefix wheel_,$(filter-out %_x86_64, $(filter-out %_i686, $(MANYLINUX_IMAGES)))): qemu-user-static

wheel_%: dist/$(PACKAGE)-$(VERSION).tar.gz
	echo "Building wheels for $(PACKAGE) $(VERSION)"
	time docker run --rm -t \
		-v $(shell pwd):/io \
		-e CFLAGS="-O3 -g0 -mtune=generic -pipe -fPIC" \
		-e LDFLAGS="$(LDFLAGS) -fPIC" \
		-e WHEELHOUSE=wheelhouse$(subst wheel_manylinux,,$@) \
		quay.io/pypa/$(subst wheel_,,$@) \
		bash -c 'for PYBIN in /opt/python/$(PYTHON_WHEEL_BUILD_VERSION)/bin; do \
		    $$PYBIN/python -V; \
		    { $$PYBIN/pip wheel -w /io/$$WHEELHOUSE /io/$< & } ; \
		    done; wait; \
		    for whl in /io/$$WHEELHOUSE/$(PACKAGE)-$(VERSION)-*-linux_*.whl; do auditwheel repair $$whl -w /io/$$WHEELHOUSE; done'
