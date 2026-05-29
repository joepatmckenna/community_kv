"""DistContext defaults — CPU-only smoke tests (no NCCL)."""

from evals.distributed import DistContext


class TestDistContext:
    def test_default_construction(self):
        d = DistContext()
        assert d.world_size == 1
        assert d.rank == 0
        assert d.local_rank == 0
        assert d.is_tp is False

    def test_from_env_no_torchrun(self, monkeypatch):
        monkeypatch.delenv("WORLD_SIZE", raising=False)
        d = DistContext.from_env()
        assert d.is_tp is False
        assert d.world_size == 1

    def test_from_env_world_size_one(self, monkeypatch):
        monkeypatch.setenv("WORLD_SIZE", "1")
        d = DistContext.from_env()
        assert d.is_tp is False
        assert d.world_size == 1

    def test_from_env_independent_instances(self, monkeypatch):
        """Two ``from_env()`` calls return independent instances —
        mutating one must not affect the other."""
        monkeypatch.delenv("WORLD_SIZE", raising=False)
        a = DistContext.from_env()
        b = DistContext.from_env()
        a.rank = 7
        assert b.rank == 0

    def test_print0_only_rank0(self, capsys):
        DistContext(rank=0).print0("hello")
        out = capsys.readouterr().out
        assert "hello" in out

        DistContext(rank=1).print0("nope")
        out = capsys.readouterr().out
        assert out == ""

    def test_barrier_noop_in_pp_mode(self):
        DistContext(is_tp=False).barrier()  # must not raise

    def test_destroy_noop_in_pp_mode(self):
        DistContext(is_tp=False).destroy()  # must not raise
